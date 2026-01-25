#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鏉垮潡杞姩妯″瀷璁粌
鐢ㄤ簬璁粌鍩轰簬鏉垮潡鐗瑰緛鐨勬澘鍧楄疆鍔ㄩ娴嬫ā鍨嬶紝缁撳悎甯傚満鐘舵€佺壒寰侊紝浼樺寲鏉垮潡閫夋嫨绛栫暐銆?

娉ㄦ剰锛氭鏂囦欢宸茶閲嶆瀯涓烘洿妯″潡鍖栫殑鏋舵瀯銆?
鎺ㄨ崘浣跨敤鏂扮殑API杩涜寮€鍙戝拰缁存姢锛?
- to/sector_model.py: 涓籄PI鍏ュ彛锛屾彁渚涙澘鍧楄疆鍔ㄦā鍨嬬殑璁粌銆侀娴嬪拰鍥炴祴鍔熻兘
- to/data_processor.py: 鏁版嵁澶勭悊妯″潡
- to/feature_processor.py: 鐗瑰緛澶勭悊妯″潡
- to/model_manager.py: 妯″瀷绠＄悊妯″潡

```

姝ゆ枃浠朵繚鐣欑敤浜庡吋瀹圭幇鏈夌殑璋冪敤鏂瑰紡銆?
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import json
import traceback
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
import copy
import pickle
import inspect
import bisect
import joblib
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import logging
import os
import time
import math
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import functools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier
from scipy.stats import spearmanr, kendalltau
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing



# 瀵煎叆logger閰嶇疆 - 淇敼涓虹浉瀵瑰鍏?
try:
    from logger_config import configure_logger
except ImportError:
    # 娣诲姞褰撳墠鐩綍鍒皊ys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from logger_config import configure_logger

# 灏嗗師鏉ョ殑logger閰嶇疆鏇挎崲涓烘柊鐨勯厤缃?
logger = configure_logger(__name__, 'train_sector_models.log')

# 鑷畾涔塉SON缂栫爜鍣紝鐢ㄤ簬澶勭悊NumPy鍜屽叾浠栫壒娈婄被鍨?
class NumpyEncoder(json.JSONEncoder):
    """鎵╁睍JSON缂栫爜鍣紝澶勭悊NumPy鍜屽叾浠栫壒娈婄被鍨?""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# 鑾峰彇椤圭洰鏍圭洰褰?
def get_project_root():
    """鑾峰彇椤圭洰鏍圭洰褰?""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

# 璺緞璁剧疆
def get_data_path(relative_path=None):
    """鑾峰彇鏁版嵁鐩綍璺緞"""
    project_root = get_project_root()
    data_path = project_root / "to" / "local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

def get_model_path():
    """鑾峰彇妯″瀷鐩綍璺緞"""
    project_root = get_project_root()
    model_path = project_root / "to" / "models" / "sector_models"
    return model_path

def ensure_dir(path):
    """纭繚鐩綍瀛樺湪"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class SectorModelTrainer:
    """鏉垮潡妯″瀷璁粌鍣?""
    
    def __init__(self, model_dir=None):
        """
        鍒濆鍖栨ā鍨嬭缁冨櫒
        
        Args:
            model_dir: 妯″瀷淇濆瓨鐩綍
        """
        self.logger = logger
        
        # 璁剧疆妯″瀷鐩綍
        self.model_dir = Path(model_dir) if model_dir else get_model_path()
        ensure_dir(self.model_dir)
        
        # 鐗瑰緛鐩綍
        self.sector_features_dir = get_data_path("sector_features")
        self.market_status_dir = get_data_path("market_status/daily")
         
        # 浜ゆ槗鏃ュ巻
        self.trading_days = []
        self._load_trading_calendar()
        
        # 娣诲姞鐗瑰緛缂撳瓨鐩綍
        self.features_cache_dir = Path(get_data_path(), "features_cache")
        ensure_dir(self.features_cache_dir)
        
        self.logger.info(f"鏉垮潡妯″瀷璁粌鍣ㄥ垵濮嬪寲瀹屾垚锛屾ā鍨嬬洰褰? {self.model_dir}")

    def _load_trading_calendar(self):
        """
        鍔犺浇浜ゆ槗鏃ュ巻
        
        Returns:
            list: 浜ゆ槗鏃ュ巻鍒楄〃锛屽鏋滃姞杞藉け璐ュ垯杩斿洖None
        """
        try:
            # 灏濊瘯浠嶱arquetDataStorage涓幏鍙?
            sys.path.append(str(get_project_root() / "to"))
            from advanced_quant_data import ParquetDataStorage
            
            storage = ParquetDataStorage()
            
            # 鑾峰彇杈冮暱鏃堕棿鑼冨洿鐨勪氦鏄撴棩鍘嗭紝浣嗕笉闇€瑕佸お鏃╃殑鏁版嵁
            # 鍙渶瑕佸寘鍚缁冩棩鏈熷拰鍥炴函鏈熷嵆鍙紝'20220101'瓒冲鏃?
            cal_df = storage.get_trading_calendar('20221001', '20250331')
            
            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_days = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_days = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                    
                self.logger.info(f"鎴愬姛鍔犺浇浜ゆ槗鏃ュ巻锛屽叡 {len(self.trading_days)} 涓氦鏄撴棩锛岃寖鍥? {self.trading_days[0]} - {self.trading_days[-1]}")
                return self.trading_days
            else:
                self.logger.warning("鑾峰彇浜ゆ槗鏃ュ巻澶辫触")
                self.trading_days = None
                return None
        except Exception as e:
            print(f"鍙戠敓閿欒: {e}")
            self.logger.error(f"鍔犺浇浜ゆ槗鏃ュ巻鍑洪敊: {e}")
            self.logger.error(traceback.format_exc())
            self.trading_days = None
            return None
       
    
    def _load_sector_features(self, sector_code, date):
        """
        鍔犺浇鐗瑰畾鏉垮潡鐨勭壒寰佹暟鎹?
        
        Args:
            sector_code: 鏉垮潡浠ｇ爜
            date: 鏃ユ湡锛屾牸寮忎负YYYYMMDD
            
        Returns:
            DataFrame: 鐗瑰緛鏁版嵁
        """
        try:
            # 鏋勫缓鏂囦欢璺緞 - 鎸夋澘鍧椾唬鐮佸垝鍒嗙洰褰?
            # 浼樺厛鍔犺浇浼樺寲鍚庣殑涓荤壒寰佹枃浠?
            main_features_path = self.sector_features_dir / sector_code / f"{sector_code}_features.parquet"
            print(f"main_features_path: {main_features_path}")
            
            if main_features_path.exists():
                # 濡傛灉涓荤壒寰佹枃浠跺瓨鍦紝鐩存帴璇诲彇骞剁瓫閫夋棩鏈?
                self.logger.warning(f"灏濊瘯浠庝富鐗瑰緛鏂囦欢鍔犺浇鏉垮潡 {sector_code} 鐨勬棩鏈?{date} 鏁版嵁")
                df = pd.read_parquet(main_features_path)
                
                # 绛涢€夋寚瀹氭棩鏈?
                if 'date' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        return date_df
                    
                if 'date_str' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        return date_df
                
                self.logger.debug(f"涓荤壒寰佹枃浠朵腑鏈壘鍒版棩鏈?{date} 鐨勬暟鎹?)
            else:
                # 灏濊瘯鍔犺浇鍗曟棩鏈熺壒寰佹枃浠?
                date_file = self.sector_features_dir / sector_code / f"{sector_code}_{date}.parquet"
                if date_file.exists():
                    self.logger.debug(f"灏濊瘯浠庡崟鏃ユ湡鏂囦欢鍔犺浇鏉垮潡 {sector_code} 鐨勬棩鏈?{date} 鏁版嵁")
                    df = pd.read_parquet(date_file)
                    return df
                
                # 灏濊瘯鍔犺浇鍘熷鍚堝苟鐗瑰緛鏂囦欢
                orig_file = self.sector_features_dir / sector_code / f"{sector_code}.parquet"
                if orig_file.exists():
                    self.logger.debug(f"灏濊瘯浠庡師濮嬪悎骞舵枃浠跺姞杞芥澘鍧?{sector_code} 鐨勬棩鏈?{date} 鏁版嵁")
                    df = pd.read_parquet(orig_file)
                    
                    # 绛涢€夋寚瀹氭棩鏈?
                    if 'date' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                        if not date_df.empty:
                            return date_df
                        
                    if 'date_str' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                        if not date_df.empty:
                            return date_df
                
                self.logger.warning(f"鏈壘鍒版澘鍧?{sector_code} 鍦ㄦ棩鏈?{date} 鐨勭壒寰佹暟鎹?)
                return None
        except Exception as e:
            self.logger.error(f"鍔犺浇鏉垮潡 {sector_code} 鍦ㄦ棩鏈?{date} 鐨勭壒寰佹暟鎹椂鍑洪敊: {e}")
            return None

    def _load_all_market_status(self):
        """
        鍔犺浇甯傚満鐘舵€佹暟鎹?
        
        Args:
            date: 鏃ユ湡锛屾牸寮忎负YYYYMMDD
            
        Returns:
            dict: 甯傚満鐘舵€佹暟鎹?
        """
        try:
            # 鍔犺浇甯傚満鐘舵€佹暟鎹?
            status_file = self.market_status_dir / "market_status.json"
            if not status_file.exists():
                self.logger.warning(f"甯傚満鐘舵€佹枃浠朵笉瀛樺湪: {status_file}")
                return None
                
            with open(status_file, "r", encoding="utf-8") as f:
                market_status_data = json.load(f)
            return market_status_data   
                
        except Exception as e:
            self.logger.error(f"鍔犺浇甯傚満鐘舵€佹暟鎹椂鍑洪敊: {e}")
            self.logger.exception(e)
            return None
    
    def train_rotation_model(self, sector_list=None, start_date=None, end_date=None, model_name="sector_rotation_model"):
        """璁粌鏉垮潡杞姩妯″瀷锛屼娇鐢ㄦ墍鏈夋澘鍧楃殑鐗瑰緛鏉ラ娴嬫澘鍧楁帓鍚?
        
        鍙傛暟:
            sector_list: 鏉垮潡浠ｇ爜鍒楄〃
            start_date: 寮€濮嬫棩鏈?
            end_date: 缁撴潫鏃ユ湡
            model_name: 妯″瀷鍚嶇О
        
        杩斿洖:
            璁粌鍚庣殑妯″瀷鍖?
        """
        logger.info(f"寮€濮嬭缁冩澘鍧楄疆鍔ㄦā鍨?..")
        
            
        # 鑾峰彇鎵€鏈夊彲鐢ㄧ殑鏉垮潡
        if sector_list is None:
            sector_list = self._get_all_sectors()
            
        if isinstance(sector_list, str) and os.path.isfile(sector_list):
            # 浠庢枃浠跺姞杞芥澘鍧楀垪琛?
            with open(sector_list, 'r', encoding='utf-8') as f:
                sector_list = [line.strip() for line in f.readlines() if line.strip()]
                
        logger.info(f"浣跨敤鏉垮潡鍒楄〃: {sector_list}, 鍏眥len(sector_list)}涓澘鍧?)
        
        # 璁剧疆璁粌鏃ユ湡鑼冨洿
        if start_date is None:
            start_date = "20200101"  # 榛樿浠?020骞村紑濮?
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        train_end_date = "20250128"
            
        # 璁剧疆棰勬祴鏃ユ湡鑼冨洿 - 2025骞?鏈?
        predict_start_date = "20250201"
        predict_end_date = "20250331"
        
        # 纭畾闇€瑕佸姞杞界殑鏁翠綋鏃ユ湡鑼冨洿
        overall_start_date = start_date
        overall_end_date = predict_end_date  # 纭繚鍔犺浇鍒伴娴嬬粨鏉熸棩鏈?
            
        logger.info(f"璁粌鏃ユ湡鑼冨洿: {start_date} 鑷?{end_date}")
        logger.info(f"棰勬祴鏃ユ湡鑼冨洿: {predict_start_date} 鑷?{predict_end_date}")
        
        # 鎵归噺鍔犺浇鎵€鏈夋澘鍧楃壒寰侊紝鍖呮嫭璁粌鏈熷拰棰勬祴鏈?
        all_features = self._batch_load_features(sector_list, overall_start_date, overall_end_date)
        
        if all_features is None or len(all_features) == 0:
            logger.error("娌℃湁鎵惧埌鏈夋晥鐨勭壒寰佹暟鎹紝璁粌缁堟")
            return None
                
        logger.info(f"鎴愬姛鍔犺浇 {len(all_features)} 涓澘鍧楃殑鐗瑰緛鏁版嵁")
        
        # 鎷嗗垎鏁版嵁涓鸿缁冮泦鍜岄娴嬮泦
        if 'date' in all_features.columns:
            # 纭繚date鍒楁槸datetime绫诲瀷
            if not pd.api.types.is_datetime64_any_dtype(all_features['date']):
                all_features['date'] = pd.to_datetime(all_features['date'])
                
            # 绛涢€夎缁冩暟鎹泦
            train_mask = ((all_features['date'] >= pd.to_datetime(start_date)) & 
                           (all_features['date'] <= pd.to_datetime(train_end_date)))
            train_features = all_features[train_mask].copy()
            
            # 绛涢€夐娴嬫暟鎹泦 (2025骞?鏈?
            predict_mask = ((all_features['date'] >= pd.to_datetime(predict_start_date)) & 
                             (all_features['date'] <= pd.to_datetime(predict_end_date)))
            predict_features = all_features[predict_mask].copy()
            
            logger.info(f"绛涢€夊嚭璁粌鏁版嵁 {len(train_features)} 鏉★紝棰勬祴鏁版嵁 {len(predict_features)} 鏉?)
        else:
            logger.error("鐗瑰緛鏁版嵁涓病鏈塪ate鍒楋紝鏃犳硶鎸夋棩鏈熺瓫閫?)
            return None
            
        # 澶勭悊璁粌鐗瑰緛锛屽噯澶囨ā鍨嬭缁?
        logger.info("寮€濮嬪噯澶囨ā鍨嬭缁冩暟鎹?..")
        
        # 璇嗗埆娑ㄨ穼鍋滅粨鏋勭浉鍏崇壒寰?
        board_structure_features = [col for col in train_features.columns if 
                                   any(col.startswith(prefix) for prefix in 
                                      ['limit_', 'broken_', 'one_word_', 'board_heat', 
                                       'market_cap_', 'consecutive_limit_', 'board_structure_',
                                       'limit_board_', 'limit_type_'])]
        
        logger.info(f"妫€娴嬪埌 {len(board_structure_features)} 涓定璺屽仠缁撴瀯鐩稿叧鐗瑰緛")
        
        # 鍑嗗鐗瑰緛鍜屾爣绛?
        # 閫夋嫨闇€瑕佺殑鐗瑰緛鍒楋紙鎺掗櫎涓嶉渶瑕佺殑鍒楋級
        exclude_cols = ['date', 'sector_code', 'sector_name', 'change_percent_1d', 
                        'change_percent_3d', 'change_percent_5d', 'change_percent_10d', 
                        'change_percent_20d', 'is_top_performer']
        
        # 鑾峰彇鎵€鏈夊彲鐢ㄧ殑鐗瑰緛鍒?
        all_feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        
        # 纭鐗瑰緛鍜岀洰鏍囧垪鏄惁瀛樺湪
        if 'is_top_performer' not in train_features.columns:
            logger.error("璁粌鏁版嵁涓病鏈夌洰鏍囧垪 'is_top_performer'")
            return None
            
        # 鎵撳嵃鐗瑰緛闆嗙殑鏋勬垚
        logger.info(f"鐗瑰緛闆嗗寘鍚€诲叡 {len(all_feature_cols)} 涓壒寰侊紝鍏朵腑 {len(board_structure_features)} 涓定璺屽仠缁撴瀯鐗瑰緛")
        
        # 鐗瑰埆鎵撳嵃娑ㄨ穼鍋滅粨鏋勭壒寰佸悕绉?
        if board_structure_features:
            logger.info(f"娑ㄨ穼鍋滅粨鏋勭壒寰? {', '.join(board_structure_features[:10])}...")
        
        # 璁板綍鐗瑰緛鏉冮噸锛屾彁楂樻定璺屽仠缁撴瀯鐗瑰緛鐨勬潈閲?
        feature_weights = {}
        for feature in all_feature_cols:
            if feature in board_structure_features:
                # 缁欐定璺屽仠缁撴瀯鐗瑰緛鏇撮珮鐨勬潈閲?
                feature_weights[feature] = 1.5
            else:
                feature_weights[feature] = 1.0
                
        # 淇濆瓨妯″瀷璁粌淇℃伅
        model_info = {
            "train_dates": {
                "start_date": start_date,
                "end_date": end_date,
                "train_end_date": train_end_date,
                "predict_start_date": predict_start_date,
                "predict_end_date": predict_end_date
            },
            "feature_counts": {
                "total_features": len(all_feature_cols),
                "board_structure_features": len(board_structure_features)
            },
            "board_structure_features": board_structure_features,
            "feature_weights": feature_weights
        }
        
        # 纭繚妯″瀷鐩綍瀛樺湪
        model_dir = os.path.join(get_model_path(), model_name)
        ensure_dir(model_dir)
        
        # 淇濆瓨妯″瀷淇℃伅
        model_info_file = os.path.join(model_dir, "model_info.json")
        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"妯″瀷淇℃伅宸蹭繚瀛樺埌 {model_info_file}")
        
        # 鍚庣画妯″瀷璁粌浠ｇ爜...

    def _batch_load_features(self, sector_list, start_date, end_date):
        """鎵归噺鍔犺浇澶氫釜鏉垮潡澶氫釜鏃ユ湡鐨勭壒寰佹暟鎹?
        
        Args:
            sector_list: 鏉垮潡鍒楄〃
            start_date: 寮€濮嬫棩鏈?
            end_date: 缁撴潫鏃ユ湡
            
        Returns:
            DataFrame: 鎵€鏈夌壒寰佹暟鎹?
        """
        # 娣诲姞缂撳瓨鏈哄埗锛岄伩鍏嶉噸澶嶅姞杞藉拰澶勭悊鏁版嵁
        cache_dir = os.path.join(get_project_root(), "to", "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 鐢熸垚缂撳瓨鏂囦欢鍚嶏紝鍩轰簬杈撳叆鍙傛暟
        sector_str = "-".join(sorted(sector_list))
        cache_key = f"{sector_str[:100]}-{start_date}-{end_date}"
        import hashlib
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        # cache_hash = "9953d2f3ca428eebeeaf18e6f986f03f"
        cache_file = os.path.join(cache_dir, f"batch_features_{cache_hash}.pkl")
        
        # 妫€鏌ユ槸鍚﹀瓨鍦ㄧ紦瀛?
        if os.path.exists(cache_file):
            try:
                self.logger.info(f"鍙戠幇缂撳瓨鏂囦欢 {cache_file}锛屽皾璇曞姞杞?..")
                with open(cache_file, 'rb') as f:
                    df_all = pickle.load(f)
                self.logger.info(f"鎴愬姛浠庣紦瀛樺姞杞界壒寰佹暟鎹紝褰㈢姸: {df_all.shape}")
                return df_all
            except Exception as e:
                self.logger.warning(f"鍔犺浇缂撳瓨鏂囦欢澶辫触: {e}锛屽皢閲嶆柊鐢熸垚鐗瑰緛")
        else:
            self.logger.info(f"缂撳瓨鏂囦欢 {cache_file} 涓嶅瓨鍦紝灏嗛噸鏂扮敓鎴愮壒寰?)
        
        # 纭繚鏈変氦鏄撴棩鍘?
        if self.trading_days is None:
            self._load_trading_calendar()
            
        # 杩囨护浜ゆ槗鏃?
        trading_days_in_range = [day for day in self.trading_days if start_date <= day <= end_date]
        self.logger.info(f"鍦?{start_date} 鑷?{end_date} 鏈熼棿鍏辨湁 {len(trading_days_in_range)} 涓氦鏄撴棩")
        
        # 鍏堝姞杞芥墍鏈夋澘鍧楀湪鏁翠釜鏃ユ湡鑼冨洿鐨勫熀纭€鏁版嵁锛屼互渚跨紦瀛?
        self.logger.info(f"寮€濮嬪姞杞?{len(sector_list)} 涓澘鍧楃殑鍩虹鏁版嵁...")
        base_features = {}
        
        # 瀹氫箟鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰佽矾寰?
        board_structure_dir = os.path.join(get_project_root(), "to", "local_data", "sector_features_board")
        self.logger.info(f"鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰佽矾寰? {board_structure_dir}")
        
        # 瀛樺偍鏉垮潡鐨勬定璺屽仠缁撴瀯鐗瑰緛
        board_structure_features = {}
        
        # 浣跨敤澶氱嚎绋嬪苟琛屽姞杞藉熀纭€鏁版嵁
        with ThreadPoolExecutor(max_workers=min(10, len(sector_list))) as executor:
            futures = {
                executor.submit(self._load_base_features, sector, start_date, end_date): sector 
                for sector in sector_list
            }
            
            for future in as_completed(futures):
                sector = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        base_features[sector] = result
                        self.logger.info(f"鎴愬姛鍔犺浇鏉垮潡 {sector} 鐨勫熀纭€鏁版嵁锛屽叡 {len(result)} 鏉¤褰?)
                    else:
                        self.logger.warning(f"鏉垮潡 {sector} 娌℃湁鎵惧埌鍩虹鏁版嵁")
                except Exception as e:
                    self.logger.error(f"鍔犺浇鏉垮潡 {sector} 鍩虹鏁版嵁鏃跺嚭閿? {e}")
        
        # 鍔犺浇鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰?
        self.logger.info("寮€濮嬪姞杞芥澘鍧楁定璺屽仠缁撴瀯鐗瑰緛鏁版嵁...")
        for sector in sector_list:
            try:
                # 鏋勫缓鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰佹枃浠惰矾寰?
                board_file = os.path.join(board_structure_dir, sector, f"{sector}_board_analysis.parquet")
                
                # 妫€鏌ユ枃浠舵槸鍚﹀瓨鍦?
                if os.path.exists(board_file):
                    # 鍔犺浇鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰?
                    board_df = pd.read_parquet(board_file)
                    
                    # 纭繚鏃ユ湡鍒楁槸datetime绫诲瀷
                    if 'date' in board_df.columns:
                        board_df['date_str'] = pd.to_datetime(board_df['date_str'])
                    elif 'date_obj' in board_df.columns:
                        board_df['date_str'] = board_df['date_obj']
                    else:
                        self.logger.warning(f"鏉垮潡 {sector} 鐨勬定璺屽仠缁撴瀯鐗瑰緛鏂囦欢娌℃湁鏃ユ湡鍒?)
                        continue
                    
                    # 绛涢€夋棩鏈熻寖鍥村唴鐨勬暟鎹?
                    mask = (board_df['date_str'] >= pd.to_datetime(start_date)) & (board_df['date_str'] <= pd.to_datetime(end_date))
                    filtered_board_df = board_df[mask].copy()
                    
                    if not filtered_board_df.empty:
                        # 璁剧疆鏃ユ湡涓虹储寮曪紝渚夸簬鍚庣画澶勭悊
                        filtered_board_df.set_index('date', inplace=True)
                        board_structure_features[sector] = filtered_board_df
                        self.logger.info(f"鎴愬姛鍔犺浇鏉垮潡 {sector} 鐨勬定璺屽仠缁撴瀯鐗瑰緛锛屽叡 {len(filtered_board_df)} 鏉¤褰?)
                    else:
                        self.logger.warning(f"鏉垮潡 {sector} 鍦ㄦ寚瀹氭棩鏈熻寖鍥村唴娌℃湁娑ㄨ穼鍋滅粨鏋勭壒寰佹暟鎹?)
                else:
                    self.logger.warning(f"鏉垮潡 {sector} 鐨勬定璺屽仠缁撴瀯鐗瑰緛鏂囦欢涓嶅瓨鍦? {board_file}")
            except Exception as e:
                self.logger.error(f"鍔犺浇鏉垮潡 {sector} 鐨勬定璺屽仠缁撴瀯鐗瑰緛鏃跺嚭閿? {e}")
        
        # 濡傛灉娌℃湁浠讳綍鏁版嵁锛屾彁鍓嶈繑鍥?
        if not base_features:
            self.logger.warning("娌℃湁鎵惧埌浠讳綍鏉垮潡鏁版嵁")
            return pd.DataFrame()
            
        self.logger.info(f"鎴愬姛鍔犺浇 {len(base_features)}/{len(sector_list)} 涓澘鍧楃殑鍩虹鏁版嵁")
        self.logger.info(f"鎴愬姛鍔犺浇 {len(board_structure_features)}/{len(sector_list)} 涓澘鍧楃殑娑ㄨ穼鍋滅粨鏋勭壒寰?)
        
        # 鎻愬墠鍔犺浇甯傚満鐜鏁版嵁
        self.logger.info("寮€濮嬪姞杞藉競鍦虹幆澧冩暟鎹?..")
        market_status_cache = self._load_all_market_status()
        self.logger.info(f"鎴愬姛鍔犺浇 {len(market_status_cache)} 涓棩鏈熺殑甯傚満鐜鏁版嵁")
        
        # 鍑嗗鎵€鏈変换鍔?- 姣忎釜鏉垮潡姣忎釜鏃ユ湡涓€涓换鍔?
        tasks = []
        for sector in base_features.keys():
            for date in trading_days_in_range:
                date_obj = pd.to_datetime(date)
                if date_obj in base_features[sector].index:
                    tasks.append((sector, date))
                    
        self.logger.info(f"鍏辨壘鍒?{len(tasks)} 涓壒寰佹瀯寤轰换鍔?)
        
        # 浣跨敤澶氱嚎绋嬪鐞嗘瘡涓澘鍧?鏃ユ湡缁勫悎
        all_features = []
        futures = {}
       
        with ThreadPoolExecutor(max_workers=min(20, len(tasks))) as executor:
            for sector, date in tasks:
                date_obj = pd.to_datetime(date)
                
                # 鑾峰彇甯傚満鐜鏁版嵁
                market_data = market_status_cache.get(date, None)
                
                # 鑾峰彇鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰?
                board_data = None
                if sector in board_structure_features and date_obj in board_structure_features[sector].index:
                    board_data = board_structure_features[sector].loc[date_obj].to_dict()
                
                # 鎻愪氦浠诲姟
                future = executor.submit(
                    self._process_features_with_layers,
                    sector_code=sector,
                    date=date_obj,
                    base_data=base_features[sector],
                    market_status=market_data,
                    board_structure=board_data
                )
                
                futures[future] = (sector, date)
            
            # 鏀堕泦缁撴灉
            for i, future in enumerate(as_completed(futures)):
                sector_code, date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # 涓嶅啀闇€瑕佸啀娆¤绠楁湭鏉ユ敹鐩婄巼锛屽洜涓哄凡缁忓湪_process_features_with_layers涓畬鎴?
                        all_features.append(result)
                    
                    # 杩涘害鏃ュ織
                    if (i+1) % 100 == 0 or (i+1) == len(tasks):
                        self.logger.info(f"宸插鐞?{i+1}/{len(tasks)} 涓壒寰佹瀯寤轰换鍔?)
                except Exception as e:
                    self.logger.error(f"澶勭悊鏉垮潡 {sector_code} 鏃ユ湡 {date} 鐨勭壒寰佹椂鍑洪敊: {e}")
        
        self.logger.info(f"鎵归噺鍔犺浇瀹屾垚锛屾垚鍔熸瀯寤?{len(all_features)} 鏉＄壒寰佽褰?)
        
        if not all_features:
            self.logger.warning("娌℃湁鐢熸垚浠讳綍鏈夋晥鐗瑰緛")
            return pd.DataFrame()
        
        # 鍚堝苟鎵€鏈夌壒寰?
        try:
            # 鍚堝苟鎵€鏈塂ataFrame
            df_all = pd.concat(all_features, ignore_index=True)
            
            # 娣诲姞鏃ユ湡瀛楁(濡傛灉娌℃湁)
            if 'date' in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all['date']):
                df_all['date'] = pd.to_datetime(df_all['date'])
                
            # 鎸夋棩鏈熸帓搴?
            if 'date' in df_all.columns:
                df_all = df_all.sort_values('date')
                
            self.logger.info(f"鐗瑰緛鏋勫缓瀹屾垚锛屽叡 {len(df_all)} 鏉¤褰曪紝{df_all.shape[1]} 涓壒寰?)
            
            # 淇濆瓨鍒扮紦瀛?
            try:
                self.logger.info(f"淇濆瓨鐗瑰緛鏁版嵁鍒扮紦瀛?{cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(df_all, f)
                self.logger.info(f"鐗瑰緛鏁版嵁缂撳瓨淇濆瓨鎴愬姛")
                
            except Exception as e:
                self.logger.warning(f"淇濆瓨鐗瑰緛鏁版嵁鍒扮紦瀛樺け璐? {e}")
            
            return df_all
            
        except Exception as e:
            print(f"鍙戠敓閿欒: {e}")
            self.logger.error(f"鍚堝苟鐗瑰緛鏁版嵁鏃跺嚭閿? {e}")
            return pd.DataFrame()
        
           

    def _load_base_features(self, sector_code, start_date, end_date):
        """鍔犺浇鍗曚釜鏉垮潡鐨勫熀纭€鐗瑰緛鏁版嵁锛堜环鏍笺€佹垚浜ら噺绛夛級
        
        Args:
            sector_code: 鏉垮潡浠ｇ爜
            start_date: 寮€濮嬫棩鏈?
            end_date: 缁撴潫鏃ユ湡
            
        Returns:
            鏉垮潡鍩虹鏁版嵁
        """
        try:
            # 鑾峰彇鏃ユ湡鑼冨洿鍐呯殑鎵€鏈変氦鏄撴棩
            if self.trading_days is None:
                self._load_trading_calendar()
                
            trading_days = [day for day in self.trading_days if start_date <= day <= end_date]
            if not trading_days:
                self.logger.warning(f"鍦▄start_date}鑷硔end_date}涔嬮棿娌℃湁鏈夋晥浜ゆ槗鏃?)
                return None
                
            # 鏋勫缓鏉垮潡涓荤壒寰佹枃浠惰矾寰?
            sector_dir = self.sector_features_dir / f"{sector_code}"
            main_features_path = sector_dir / f"{sector_code}_features.parquet"
            
            # 濡傛灉涓荤壒寰佹枃浠朵笉瀛樺湪锛屽皾璇曚娇鐢ㄥ悎骞剁壒寰佹枃浠?
            if not main_features_path.exists():
                merged_file_path = sector_dir / f"{sector_code}.parquet"
                if merged_file_path.exists():
                    main_features_path = merged_file_path
                else:
                    self.logger.warning(f"鏉垮潡 {sector_code} 娌℃湁涓荤壒寰佹枃浠舵垨鍚堝苟鐗瑰緛鏂囦欢")
                    return None
            
            # 璇诲彇鐗瑰緛鏂囦欢
            self.logger.info(f"浠?{main_features_path} 璇诲彇 {sector_code} 鏉垮潡浠?{start_date} 鍒?{end_date} 鐨勭壒寰佹暟鎹?)
            
            try:
                df = pd.read_parquet(main_features_path)
                
                # 纭繚鏃ユ湡鍒楀瓨鍦?
                if 'date' not in df.columns:
                    self.logger.warning(f"鐗瑰緛鏂囦欢 {main_features_path} 娌℃湁date鍒?)
                    return None
                
                # 灏嗘棩鏈熷垪杞崲涓篸atetime瀵硅薄渚夸簬姣旇緝
                df['date_str'] = pd.to_datetime(df['date_str'])
                
                # 绛涢€夋棩鏈熻寖鍥村唴鐨勬暟鎹?
                mask = (df['date_str'] >= pd.to_datetime(start_date)) & (df['date_str'] <= pd.to_datetime(end_date))
                filtered_df = df[mask].copy()
                
                if filtered_df.empty:
                    self.logger.warning(f"鏉垮潡 {sector_code} 鍦?{start_date} 鑷?{end_date} 鏈熼棿娌℃湁鏁版嵁")
                    return None
                
                # 璁剧疆date涓虹储寮曚互渚垮悗缁鐞?
                filtered_df.set_index('date', inplace=True)
                
                self.logger.info(f"鎴愬姛鍔犺浇 {sector_code} 鏉垮潡鐗瑰緛锛屽叡 {len(filtered_df)} 鏉¤褰?)
                return filtered_df
                
            except Exception as e:
                print(f"鍙戠敓閿欒: {e}")
                self.logger.error(f"璇诲彇 {main_features_path} 鍑洪敊: {e}")
                return None
                
                
        except Exception as e:
            print(f"鍙戠敓閿欒: {e}")
            self.logger.error(f"鍔犺浇鏉垮潡 {sector_code} 鍩虹鏁版嵁鏃跺彂鐢熼敊璇? {e}")
            return None


    def _process_features_with_layers(self, sector_code, date, base_data, market_status, board_structure):
        """浣跨敤涓夊眰鐗瑰緛浣撶郴澶勭悊鍗曚釜鏉垮潡鍗曚釜鏃ユ湡鐨勭壒寰?
        
        Args:
            sector_code: 鏉垮潡浠ｇ爜
            date: 鏃ユ湡
            base_data: 鍩虹灞傛暟鎹?
            market_status: 甯傚満鐜灞傛暟鎹?
            board_structure: 鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰?
        Returns:
            澶勭悊鍚庣殑鐗瑰緛鏁版嵁
        """
        try:
            # 纭繚鍩虹鏁版嵁涓寘鍚綋鍓嶆棩鏈?
            if base_data is None or date not in base_data.index:
                return None
                
            # 鑾峰彇褰撴棩鏁版嵁
            current_day = base_data.loc[date]
            
            # 鏋勫缓鐗瑰緛瀛楀吀
            features = {}
            
            # 1. 鍩虹灞傜壒寰?- 浠呬繚鐣欓噸瑕佺殑鍩虹鎸囨爣
            # ------------------------------------------------------------
            features["sector_code"] = sector_code
            features["date"] = date
            
            # 娣诲姞鍩虹浠锋牸绫绘寚鏍?
            for field in ["close", "change_percent", "volume", "amount", "turnover_rate", "amplitude"]:
                if field in current_day:
                    features[field] = current_day[field]
            
            # 娣诲姞鑲＄エ缁熻淇℃伅
            for field in ["stock_count", "up_count", "down_count", "flat_count"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 鎻愬彇涓婃定/涓嬭穼姣斾緥
            if all(f in current_day for f in ["up_count", "stock_count"]) and current_day["stock_count"] > 0:
                features["up_ratio"] = current_day["up_count"] / current_day["stock_count"]
                
            if all(f in current_day for f in ["down_count", "stock_count"]) and current_day["stock_count"] > 0:
                features["down_ratio"] = current_day["down_count"] / current_day["stock_count"]
            
            # ===== 鏂板锛氳祫閲戞祦鍔ㄧ浉鍏崇壒寰?=====
            # ------------------------------------------------------------
            # 璧勯噾娴佸姩鍔ㄨ兘鎸囨爣锛堢幇鏈夌壒寰佸熀纭€涓婃瀯寤猴級
            if "amount" in current_day and base_data is not None:
                try:
                    # 璁＄畻5鏃ュ钩鍧囨垚浜ら
                    date_idx = base_data.index.get_loc(date)
                    if date_idx >= 4:  # 纭繚鏈夎冻澶熺殑鍘嗗彶鏁版嵁
                        historical_dates = base_data.index[date_idx-4:date_idx+1]
                        historical_amounts = base_data.loc[historical_dates, "amount"]
                        avg_amount_5d = historical_amounts.mean()
                        if avg_amount_5d > 0:
                            features["fund_flow_momentum"] = current_day["amount"] / avg_amount_5d
                except Exception as e:
                    self.logger.debug(f"璁＄畻fund_flow_momentum鏃跺嚭閿? {e}")
            
            # 鐩稿鎴愪氦閲忓瘑闆嗗害锛堣　閲忚祫閲戣仛闆嗙▼搴︼級
            if "volume" in current_day and "volatility_20d" in current_day and current_day["volatility_20d"] > 0:
                features["volume_intensity"] = current_day["volume"] / current_day["volatility_20d"]
            
            # 涓婃定鎴愪氦閲忓崰姣旓紙琛￠噺璧勯噾璐ㄩ噺锛?
            if all(k in current_day for k in ["up_count", "volume", "stock_count"]) and "volume" in base_data.columns:
                try:
                    date_idx = base_data.index.get_loc(date)
                    if date_idx >= 4:  # 纭繚鏈夎冻澶熺殑鍘嗗彶鏁版嵁
                        historical_dates = base_data.index[date_idx-4:date_idx+1]
                        avg_volume_5d = base_data.loc[historical_dates, "volume"].mean()
                        if avg_volume_5d > 0 and current_day["stock_count"] > 0:
                            features["up_volume_ratio"] = (current_day["up_count"] * current_day["volume"]) / (current_day["stock_count"] * avg_volume_5d)
                except Exception as e:
                    self.logger.debug(f"璁＄畻up_volume_ratio鏃跺嚭閿? {e}")
            
            # 2. 鎶€鏈寚鏍囧眰 - 鐩存帴浣跨敤棰勮绠楃殑鎶€鏈寚鏍?
            # ------------------------------------------------------------
            # 娣诲姞鍧囩嚎鏁版嵁
            # for ma in ["ma5", "ma10", "ma20", "ma30", "ma60", "ma120"]:
            #     if ma in current_day:
            #         features[ma] = current_day[ma]
                    
            # 娣诲姞鍧囩嚎姣旂巼鎸囨爣
            for ratio in ["ma5_10_ratio", "ma5_20_ratio", "ma10_20_ratio", 
                         "ma20_60_ratio", "ma60_120_ratio", "price_ma5_ratio", 
                         "price_ma20_ratio", "price_ma60_ratio"]:
                if ratio in current_day:
                    features[ratio] = current_day[ratio]
                    
            # 娣诲姞浠锋牸浣嶇疆鎸囨爣
            for pos in ["price_above_ma5", "price_above_ma20", "price_above_ma60"]:
                if pos in current_day:
                    features[pos] = current_day[pos]
                    
            # 娣诲姞瓒嬪娍鎸囨爣
            for trend in ["trend_direction_score", "ma_bull_trend", "ma_bear_trend", 
                         "ma_neutral_trend", "ma5_slope", "ma20_slope"]:
                if trend in current_day:
                    features[trend] = current_day[trend]
                    
            # 娣诲姞鍔ㄩ噺鎸囨爣
            for mom in ["momentum_5d", "momentum_10d", "momentum_20d", "change_pct"]:
                if mom in current_day:
                    features[mom] = current_day[mom]
                    
            # # 娣诲姞RSI鎸囨爣
            # for rsi in ["rsi6", "rsi14", "rsi_overbought", "rsi_oversold", "rsi_neutral"]:
            #     if rsi in current_day:
            #         features[rsi] = current_day[rsi]
                    
            # # 娣诲姞MACD鎸囨爣
            # for macd in ["macd_line", "macd_signal", "macd_histogram", "macd_trend"]:
            #     if macd in current_day:
            #         features[macd] = current_day[macd]
                    
            # 娣诲姞娉㈠姩鐜囨寚鏍?
            for vol in ["volatility", "volatility_5d", "volatility_10d", 
                       "volatility_20d", "volatility_60d", "volatility_ratio"]:
                if vol in current_day:
                    features[vol] = current_day[vol]
                    
            # 娣诲姞鎴愪氦閲忔寚鏍?
            for vol in ["volume_ratio_5", "volume_ratio_10", "volume_ratio_20", 
                       "volume_ratio_60", "volume_change_ratio", "volume_trend"]:
                if vol in current_day:
                    features[vol] = current_day[vol]
            
            # ===== 鏂板锛氱敓鍛藉懆鏈熸寚鏍?=====
        
            
            # 鐑害鎸佺画鎬ф寚鏍?
            if "consecutive_up_days" in current_day and "indicator_consensus" in features:
                features["heat_persistence"] = current_day["consecutive_up_days"] * (features["indicator_consensus"] + 1) / 2
            
            # 鏉垮潡娲昏穬搴?
            if "turnover_rate" in current_day and "limit_up_ratio" in features:
                features["sector_vitality"] = (current_day["turnover_rate"] * features["limit_up_ratio"]) ** 0.5
            
            
            # ===== 澧炲己鎶€鏈寚鏍囩壒寰?=====
            # RSI鍜孧ACD缁勫悎澧炲己鐗瑰緛
            if all(key in current_day for key in ["rsi14", "macd_line", "macd_signal"]):
                # RSI鍜孧ACD閲戝弶/姝诲弶鍗忓悓淇″彿
                features["rsi_macd_bull_signal"] = int(current_day["rsi14"] > 50 and current_day["macd_line"] > current_day["macd_signal"] and current_day["macd_line"] > 0)
                features["rsi_macd_bear_signal"] = int(current_day["rsi14"] < 50 and current_day["macd_line"] < current_day["macd_signal"] and current_day["macd_line"] < 0)
                
                # RSI瓒呬拱/瓒呭崠鍖洪棿鐨凪ACD淇″彿(鏇村彲闈犵殑鍙嶈浆淇″彿)
                if "rsi_overbought" in current_day and current_day["rsi_overbought"] == 1:
                    features["overbought_macd_bearish"] = int(current_day["macd_line"] < current_day["macd_signal"])
                if "rsi_oversold" in current_day and current_day["rsi_oversold"] == 1:
                    features["oversold_macd_bullish"] = int(current_day["macd_line"] > current_day["macd_signal"])
            
            # 鍔ㄩ噺涓庢尝鍔ㄧ巼缁勫悎鐗瑰緛
            if all(key in current_day for key in ["momentum_20d", "volatility_20d"]) and current_day["volatility_20d"] > 0:
                # 椋庨櫓璋冩暣鍔ㄩ噺姣旂巼(绫讳技澶忔櫘姣旂巼鎬濇兂)
                features["momentum_volatility_ratio"] = current_day["momentum_20d"] / current_day["volatility_20d"]
                
            # 鎴愪氦閲忎笌浠锋牸/鍔ㄩ噺鍏崇郴
            if all(key in current_day for key in ["volume_ratio_5", "momentum_5d"]):
                # 閲忎环鍗忓悓鎸囨爣
                features["volume_momentum_confirm"] = int(
                    (current_day["volume_ratio_5"] > 1.2 and current_day["momentum_5d"] > 0) or 
                    (current_day["volume_ratio_5"] < 0.8 and current_day["momentum_5d"] < 0)
                )
                # 閲忎环鑳岀鎸囨爣(鍙兘鐨勫弽杞俊鍙?
                features["volume_momentum_divergence"] = int(
                    (current_day["volume_ratio_5"] > 1.2 and current_day["momentum_5d"] < 0) or 
                    (current_day["volume_ratio_5"] < 0.8 and current_day["momentum_5d"] > 0)
                )
            
            # 鎶€鏈寚鏍囦竴鑷存€у己搴?
            trend_signals = []
            if "macd_line" in current_day and "macd_signal" in current_day:
                trend_signals.append(1 if current_day["macd_line"] > current_day["macd_signal"] else -1)
            if "rsi14" in current_day:
                trend_signals.append(1 if current_day["rsi14"] > 50 else -1)
            if "momentum_10d" in current_day:
                trend_signals.append(1 if current_day["momentum_10d"] > 0 else -1)
            
            if trend_signals:
                # 鎶€鏈寚鏍囦竴鑷寸▼搴?-1鍒?涔嬮棿锛?琛ㄧず鎵€鏈夋寚鏍囦竴鑷寸湅澶?
                features["indicator_consensus"] = sum(trend_signals) / len(trend_signals)
                # 寮虹儓鐪嬪/鐪嬬┖淇″彿(鎵€鏈夋寚鏍囦竴鑷?
                features["strong_bull_consensus"] = int(sum(trend_signals) == len(trend_signals))
                features["strong_bear_consensus"] = int(sum(trend_signals) == -len(trend_signals))
                
            # 鎴樻棤涓嶅厠鎸囨爣(MACD涓嶳SI瀹岀編缁撳悎)
            if all(key in features for key in ["rsi6", "rsi14", "macd_line", "macd_signal"]):
                # RSI鐭嚎绐佺牬闀跨嚎 + MACD閲戝弶缁勫悎淇″彿
                features["winning_bull_signal"] = int(current_day["rsi6"] > current_day["rsi14"] 
                                                     and current_day["macd_line"] > current_day["macd_signal"]
                                                     and current_day["rsi6"] > 50)
                # RSI鐭嚎璺岀牬闀跨嚎 + MACD姝诲弶缁勫悎淇″彿
                features["winning_bear_signal"] = int(current_day["rsi6"] < current_day["rsi14"] 
                                                     and current_day["macd_line"] < current_day["macd_signal"]
                                                     and current_day["rsi6"] < 50)
            
                    
            # 娣诲姞瓒嬪娍寮哄害鍜岃繛缁ぉ鏁版寚鏍?
            for field in ["trend_strength", "consecutive_up_days", "consecutive_down_days", 
                         "swing_pattern"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 娣诲姞鏀拺闃诲姏浣嶆寚鏍?
            for field in ["support_distance_pct", 
                         "resistance_distance_pct"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 娣诲姞鎶€鏈瘎鍒嗗拰甯傚満鎯呯华鎸囨爣
            for field in ["tech_score", "market_sentiment", "signal_strength"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # ===== 鏂板锛氭坊鍔犳定璺屽仠鏉跨粨鏋勭壒寰?=====
            # ------------------------------------------------------------
            # 璋冪敤鏂版柟娉曟坊鍔犳澘鍧楁定璺屽仠缁撴瀯鐗瑰緛锛屼紶閫抌oard_structure鍙傛暟
            features = self._add_board_structure_features(features, current_day, sector_code, board_structure)
                    
            # 娣诲姞娑ㄨ穼鍋滅浉鍏虫寚鏍?
            for field in ["limit_up_ratio", "limit_down_ratio", "near_limit_up_ratio", 
                         "limit_open_ratio"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 娣诲姞鎺掑悕鐩稿叧鎸囨爣
            for field in ["ten_day_rise_count", "ten_day_rise_ratio", "daily_rise_count", 
                         "daily_rise_ratio", "daily_fall_count", "daily_fall_ratio", 
                         "daily_amount_count", "daily_amount_ratio", "ten_day_rise_weighted"]:
                if field in current_day:
                    features[field] = current_day[field]
            
            # 3. 甯傚満鐜灞?- 鐩存帴浣跨敤鎻愪緵鐨勫競鍦虹姸鎬佹暟鎹?
            # ------------------------------------------------------------
            # 鏄惧紡璁剧疆market_group鐗瑰緛锛屾棤璁烘槸鍚︽湁甯傚満鐘舵€佹暟鎹?
            if market_status:
                # 娣诲姞甯傚満绫诲瀷鍜屽懆鏈熸寚鏍?
                features["market_type"] = market_status.get("market_type", 0)
                features["market_cycle"] = market_status.get("market_cycle", 0)
                features["market_trend_strength"] = market_status.get("trend_strength", 0)
                
                # 鏄惧紡璁剧疆market_group鐗瑰緛锛屽皢market_type鏄犲皠鍒癰ull/bear/neutral/extreme
                market_type = market_status.get("market_type", 0)
                trend_strength = market_status.get("trend_strength", 0)
                
                if isinstance(market_type, (int, float)):
                    # 鑰冭檻甯傚満绫诲瀷鍜岃秼鍔垮己搴︾殑缁煎悎鏄犲皠
                    # 1. 鏄庣‘鐨勪笂娑ㄨ秼鍔?(寮哄澶村競鍦哄拰娓╁拰澶氬ご甯傚満)
                    if market_type in [11, 12, 13, 21, 22, 23]:
                        features["market_group"] = 1  # 鐗涘競
                    # 2. 鏄庣‘鐨勪笅璺岃秼鍔?(寮虹┖澶村競鍦哄拰娓╁拰绌哄ご甯傚満)
                    elif market_type in [41, 42, 43, 51, 52, 53]:
                        features["market_group"] = 2  # 鐔婂競
                    # 3. 鏋佺甯傚満鐘舵€?(甯傚満鎭愭厡鎴栫媯鐑?
                    elif market_type in [54, 55, 56, 57, 58, 59] or trend_strength > 85:
                        features["market_group"] = 4  # 鏋佺甯傚満
                    # 4. 钃勫娍缁撴瀯鎴栨俯鍜岃秼鍔胯浆鎹?
                    elif market_type in [31, 32, 33, 34, 35, 36]:
                        features["market_group"] = 3  # 闇囪崱甯傚満
                    # 5. 鍏朵粬涓嶆槑纭姸鎬?
                    else:
                        features["market_group"] = 3  # 榛樿涓洪渿鑽″競鍦?
                else:
                    # 榛樿涓簄eutral甯傚満
                    features["market_group"] = 3
            else:
                # 濡傛灉娌℃湁甯傚満鐘舵€佹暟鎹紝浣跨敤榛樿鍊?闇囪崱甯傚満)
                features["market_type"] = 33  # 33琛ㄧず闇囪崱甯傚満绫诲瀷
                features["market_cycle"] = 0
                features["market_trend_strength"] = 0
                features["market_group"] = 3  # 3琛ㄧず闇囪崱甯傚満
            
            # 娣诲姞鍒涗笟鏉挎寚鏁版暟鎹?399006.SZ)
            if "individual_status" in market_status and "399006.SZ" in market_status["individual_status"]:
                cyb_data = market_status["individual_status"]["399006.SZ"]
                
                # 鑾峰彇鍘熷鐨勭浉瀵规定璺屽箙鐗瑰緛
                if "change_percent" in cyb_data:
                    features["relative_cyb_change_percent"] = features["change_percent"] - cyb_data["change_percent"]
                
                # ===== 甯傚満涓庢澘鍧楃殑鐩稿鐗瑰緛澧炲己 =====
                # 鎻愬彇甯傚満鎶€鏈寚鏍囩敤浜庣浉瀵圭壒寰佽绠?
                if "technical_data" in cyb_data:
                    tech_data = cyb_data["technical_data"]
                    
                    # 1. 鍔ㄩ噺涓庡競鍦虹浉瀵圭壒寰?
                    if "momentum_10d" in current_day and "rsi6" in tech_data:
                        features["momentum_vs_market_rsi"] = current_day["momentum_10d"] / (tech_data["rsi6"] + 1e-8)
                    
                    # 2. 娉㈠姩鐜囦笌甯傚満鐩稿鐗瑰緛
                    if "volatility_20d" in current_day and "bb_width" in tech_data:
                        features["volatility_vs_market"] = current_day["volatility_20d"] / (tech_data["bb_width"] + 1e-8)
                    
                    # 3. RSI涓庡競鍦虹浉瀵圭壒寰?
                    if "rsi14" in current_day and "rsi12" in tech_data:
                        features["rsi_vs_market"] = current_day["rsi14"] / (tech_data["rsi12"] + 1e-8)
                        # RSI宸€肩壒寰?鐩稿寮哄急)
                        features["rsi_diff_market"] = current_day["rsi14"] - tech_data["rsi12"]
                    
                    # 4. MACD涓庡競鍦虹浉瀵圭壒寰?
                    if "macd_line" in current_day and "macd" in tech_data:
                        features["macd_vs_market"] = current_day["macd_line"] / (tech_data["macd"] + 1e-8)
                        # MACD宸€肩壒寰?
                        features["macd_diff_market"] = current_day["macd_line"] - tech_data["macd"]
                    
                    # 5. 鍧囩嚎浣嶇疆鐩稿鐗瑰緛
                    if "ma5_10_ratio" in current_day and "ma5" in tech_data and "ma10" in tech_data:
                        market_ma_ratio = tech_data["ma5"] / (tech_data["ma10"] + 1e-8)
                        features["ma_ratio_vs_market"] = current_day["ma5_10_ratio"] / (market_ma_ratio + 1e-8)
                
                # 6. 鐩堜簭姣旂浉瀵圭壒寰?
                if "profit_loss_ratio_14" in cyb_data and "support_distance_pct" in current_day and "resistance_distance_pct" in current_day:
                    if current_day["resistance_distance_pct"] != 0:
                        sector_pl_ratio = current_day["support_distance_pct"] / (current_day["resistance_distance_pct"] + 1e-8)
                        features["profit_loss_ratio_vs_market"] = sector_pl_ratio / (cyb_data["profit_loss_ratio_14"] + 1e-8)
                
                # ===== 甯傚満鐘舵€佹潯浠剁壒寰?=====
                # 鏍规嵁涓嶅悓甯傚満绫诲瀷鍒涘缓鏉′欢鐗瑰緛
                market_group = features["market_group"]
                
                # 1. 鍔ㄩ噺鏉′欢鐗瑰緛
                for mom_period in ["5d", "10d", "20d"]:
                    mom_key = f"momentum_{mom_period}"
                    if mom_key in current_day:
                        # 鐗涘競鍔ㄩ噺鏇撮噸瑕?
                        if market_group == 1:  # 鐗涘競
                            features[f"{mom_key}_in_bull"] = current_day[mom_key] * 1.2
                        # 鐔婂競璐熷姩閲忔洿閲嶈
                        elif market_group == 2:  # 鐔婂競
                            if current_day[mom_key] < 0:
                                features[f"{mom_key}_in_bear"] = current_day[mom_key] * 1.3
                            else:
                                features[f"{mom_key}_in_bear"] = current_day[mom_key] * 0.8
                        # 鏋佺甯傚満鍔ㄩ噺鏇存晱鎰?
                        elif market_group == 4:  # 鏋佺甯傚満
                            if current_day[mom_key] > 0:
                                features[f"{mom_key}_in_extreme"] = current_day[mom_key] * 1.5  # 鏋佺甯傚満鏀惧ぇ鍔ㄩ噺褰卞搷
                            else:
                                features[f"{mom_key}_in_extreme"] = current_day[mom_key] * 1.8  # 鏋佺甯傚満鏀惧ぇ璐熷姩閲忓奖鍝?
                        # 闇囪崱甯傚姩閲忛€備腑
                        else:  # 闇囪崱甯?
                            features[f"{mom_key}_in_neutral"] = current_day[mom_key]
                
                # 2. RSI鏉′欢鐗瑰緛
                if "rsi14" in current_day:
                    rsi_val = current_day["rsi14"]
                    # 鐗涘競RSI楂樹綅鏇村彲闈?
                    if market_group == 1:  # 鐗涘競
                        features["rsi_bull_signal"] = 1 if rsi_val > 60 else 0
                        features["rsi_bull_overbought"] = 1 if rsi_val > 80 else 0
                    # 鐔婂競RSI浣庝綅鏇村彲闈?
                    elif market_group == 2:  # 鐔婂競
                        features["rsi_bear_signal"] = 1 if rsi_val < 40 else 0
                        features["rsi_bear_oversold"] = 1 if rsi_val < 20 else 0
                    # 鏋佺甯傚満RSI鏋佸€兼洿鏈夋剰涔?
                    elif market_group == 4:  # 鏋佺甯傚満
                        features["rsi_extreme_high"] = 1 if rsi_val > 85 else 0
                        features["rsi_extreme_low"] = 1 if rsi_val < 15 else 0
                        # 鏋佺甯傚満涓殑鍙嶈浆淇″彿
                        features["rsi_extreme_reversal"] = 1 if (20 <= rsi_val <= 30) or (70 <= rsi_val <= 80) else 0
                    # 闇囪崱甯備腑鍊煎尯闂存洿閲嶈
                    else:  # 闇囪崱甯?
                        features["rsi_neutral_middle"] = 1 if 40 <= rsi_val <= 60 else 0
                
                # 3. MACD鏉′欢鐗瑰緛
                if all(k in current_day for k in ["macd_line", "macd_signal"]):
                    macd_line = current_day["macd_line"]
                    macd_signal = current_day["macd_signal"]
                    
                    # 鐗涘競鍏虫敞閲戝弶
                    if market_group == 1:  # 鐗涘競
                        features["macd_bull_cross"] = 1 if macd_line > macd_signal and macd_line > 0 else 0
                    # 鐔婂競鍏虫敞姝诲弶
                    elif market_group == 2:  # 鐔婂競
                        features["macd_bear_cross"] = 1 if macd_line < macd_signal and macd_line < 0 else 0
                    # 鏋佺甯傚満鍏虫敞鍋忕绋嬪害
                    elif market_group == 4:  # 鏋佺甯傚満
                        # 璁＄畻MACD鍋忕绋嬪害
                        deviation = abs(macd_line - macd_signal) / (abs(macd_signal) + 1e-8)
                        features["macd_extreme_deviation"] = deviation  # 鍋忕绋嬪害
                        # 鏋佺甯傚満涓殑鑳岀淇″彿(鍙兘鐨勫弽杞?
                        if "momentum_5d" in current_day:
                            # MACD涓庣煭鏈熷姩閲忔柟鍚戠浉鍙嶆槸绉嶈儗绂?
                            features["macd_extreme_divergence"] = 1 if (macd_line > 0 and current_day["momentum_5d"] < 0) or (macd_line < 0 and current_day["momentum_5d"] > 0) else 0
                    # 闇囪崱甯傚叧娉ㄩ浂杞寸┛瓒?
                    else:  # 闇囪崱甯?
                        features["macd_neutral_zero_cross"] = 1 if (macd_line > 0 and macd_signal < 0) or (macd_line < 0 and macd_signal > 0) else 0
                
                # 4. 鎴愪氦閲忔潯浠剁壒寰?
                if "volume_ratio_5" in current_day:
                    vol_ratio = current_day["volume_ratio_5"]
                    # 鐗涘競鏀鹃噺涓婃定鏇村ソ
                    if market_group == 1:  # 鐗涘競
                        features["volume_bull_signal"] = 1 if vol_ratio > 1.2 else 0
                    # 鐔婂競缂╅噺鏇村ソ
                    elif market_group == 2:  # 鐔婂競
                        features["volume_bear_signal"] = 1 if vol_ratio < 0.8 else 0
                    # 鏋佺甯傚満鍏虫敞鎴愪氦閲忓紓甯?
                    elif market_group == 4:  # 鏋佺甯傚満 
                        # 鏋佺鏀鹃噺鎴栨瀬绔缉閲忛兘鏄噸瑕佷俊鍙?
                        features["volume_extreme_high"] = 1 if vol_ratio > 1.8 else 0
                        features["volume_extreme_low"] = 1 if vol_ratio < 0.5 else 0
                    # 闇囪崱甯傜ǔ瀹氭垚浜ら噺鏇村ソ
                    else:  # 闇囪崱甯?
                        features["volume_neutral_signal"] = 1 if 0.8 <= vol_ratio <= 1.2 else 0
                
                # 5. 澶嶅悎浜や簰鏉′欢鐗瑰緛
                # 鐗涘競鏉′欢涓嬬殑閲忎环閰嶅悎
                if market_group == 1 and "momentum_5d" in current_day and "volume_ratio_5" in current_day:
                    features["bull_momentum_volume"] = current_day["momentum_5d"] * current_day["volume_ratio_5"] if current_day["momentum_5d"] > 0 else 0
                
                # 鐔婂競鏉′欢涓嬬殑鍙嶅脊鏈轰細
                elif market_group == 2 and "rsi14" in current_day and "consecutive_down_days" in current_day:
                    # 鐔婂競涓秴璺屽弽寮逛俊鍙?
                    features["bear_oversold_bounce"] = 1 if current_day["rsi14"] < 30 and current_day["consecutive_down_days"] >= 3 else 0
                
                # 鏋佺甯傚満鏉′欢涓嬬殑鍙嶈浆淇″彿
                elif market_group == 4:
                    # 1. 鏋佺甯傚満涓嬬殑RSI涓庢尝鍔ㄧ巼缁勫悎鐗瑰緛
                    if all(k in current_day for k in ["rsi14", "volatility_20d"]):
                        # 楂樻尝鍔ㄧ巼涓嬬殑RSI鍙嶈浆淇″彿鏇村彲闈?
                        if current_day["volatility_20d"] > 0.03:  # 楂樻尝鍔ㄧ巼鐜
                            if current_day["rsi14"] < 20:
                                features["extreme_oversold_high_vol"] = 1
                            elif current_day["rsi14"] > 80:
                                features["extreme_overbought_high_vol"] = 1
                    
                    # 2. 鏋佺甯傚満涓嬬殑閲忎环鑳岀淇″彿
                    if all(k in current_day for k in ["volume_ratio_5", "momentum_5d"]):
                        # 浠峰崌閲忕缉鎴栦环璺岄噺澧為兘鏄綔鍦ㄧ殑鍙嶈浆淇″彿
                        is_divergent = ((current_day["momentum_5d"] > 0 and current_day["volume_ratio_5"] < 0.8) or 
                                       (current_day["momentum_5d"] < 0 and current_day["volume_ratio_5"] > 1.2))
                        features["extreme_volume_price_divergence"] = 1 if is_divergent else 0
                
                # 闇囪崱甯傛潯浠朵笅鐨勬尝鍔ㄤ俊鍙?
                elif market_group == 3 and "volatility_5d" in current_day:
                    features["neutral_volatility_signal"] = current_day["volatility_5d"] / 0.02  # 鏍囧噯鍖?
                
                # ===== 鏂板锛氭澘鍧楅棿鍏崇郴鐗瑰緛 =====
                # ------------------------------------------------------------
                # 鏉垮潡鐩稿叧鎬х壒寰侊紙涓庡垱涓氭澘鎸囨暟鐨勭浉鍏虫€э級
                if "technical_data" in cyb_data and "close" in base_data.columns:
                    try:
                        # 璁＄畻鏉垮潡涓庡垱涓氭澘鐨勬粴鍔ㄧ浉鍏虫€?
                        date_idx = base_data.index.get_loc(date)
                        if date_idx >= 19:  # 纭繚鏈夎冻澶熺殑鍘嗗彶鏁版嵁(鑷冲皯20澶?
                            historical_dates = base_data.index[date_idx-19:date_idx+1]
                            
                            if all(d in base_data.index for d in historical_dates):
                                # 鑾峰彇鏉垮潡鍘嗗彶浠锋牸
                                sector_prices = base_data.loc[historical_dates, "close"]
                                
                                # 鑾峰彇鍒涗笟鏉垮巻鍙蹭环鏍?(鍋囪鍒涗笟鏉挎暟鎹寜鏃ユ湡绱㈠紩)
                                cyb_prices = []
                                for d in historical_dates:
                                    d_str = d.strftime('%Y%m%d')
                                    if "individual_status" in market_status and "399006.SZ" in market_status["individual_status"]:
                                        cyb_data_for_date = market_status["individual_status"]["399006.SZ"]
                                        if "close" in cyb_data_for_date:
                                            cyb_prices.append(cyb_data_for_date["close"])
                                
                                # 濡傛灉鏈夎冻澶熺殑鏁版嵁鐐癸紝璁＄畻鐩稿叧鎬?
                                if len(cyb_prices) >= 10:
                                    # 璁＄畻鏀剁泭鐜?
                                    sector_returns = sector_prices.pct_change().dropna()
                                    cyb_returns = pd.Series(cyb_prices).pct_change().dropna()
                                    
                                    # 纭繚闀垮害涓€鑷?
                                    min_len = min(len(sector_returns), len(cyb_returns))
                                    if min_len >= 10:
                                        corr = sector_returns[-min_len:].corr(cyb_returns[-min_len:])
                                        features["correlation_with_cyb"] = corr
                    except Exception as e:
                        self.logger.debug(f"璁＄畻鏉垮潡涓庡垱涓氭澘鐩稿叧鎬ф椂鍑洪敊: {e}")
                
                # 璧勯噾娴佽浆鐗瑰緛锛堟澘鍧楄祫閲戝崰姣斿彉鍖栵級
                if "amount" in current_day and "total_market_amount" in market_status:
                    try:
                        market_total_amount = market_status["total_market_amount"]
                        if market_total_amount > 0:
                            # 褰撳墠鏉垮潡鎴愪氦棰濆崰甯傚満鎬绘垚浜ら鐨勬瘮渚?
                            current_share = current_day["amount"] / market_total_amount
                            
                            # 灏濊瘯鑾峰彇5澶╁墠鐨勫崰姣?
                            date_idx = base_data.index.get_loc(date)
                            if date_idx >= 5 and "total_market_amount_history" in market_status:
                                past_date = base_data.index[date_idx - 5]
                                past_amount = base_data.loc[past_date, "amount"]
                                past_market_amount = market_status["total_market_amount_history"].get(past_date.strftime('%Y%m%d'), 0)
                                
                                if past_market_amount > 0:
                                    past_share = past_amount / past_market_amount
                                    features["fund_flow_share_change"] = current_share - past_share
                    except Exception as e:
                        self.logger.debug(f"璁＄畻璧勯噾娴佽浆鐗瑰緛鏃跺嚭閿? {e}")
        
            
            # 鍒涘缓鏁版嵁甯у苟杩斿洖
            return pd.DataFrame([features])
        except Exception as e:
            print(f"鍙戠敓閿欒: {e}")
            self.logger.error(f"澶勭悊鏉垮潡 {sector_code} 鏃ユ湡 {date} 鐗瑰緛鏃堕敊璇? {str(e)}")
            traceback.print_exc()
            return None

    def _get_all_sectors(self):
        """鑾峰彇鎵€鏈夊彲鐢ㄧ殑鏉垮潡鍒楄〃
        
        杩斿洖:
            鏉垮潡浠ｇ爜鍒楄〃
        """
        try:
            # 棣栧厛灏濊瘯鑾峰彇鏈湴瀛樺偍鐨勬澘鍧楀垪琛?
            sector_dirs = [d for d in os.listdir(self.sector_features_dir) 
                         if os.path.isdir(os.path.join(self.sector_features_dir, d))]
            
            if sector_dirs:
                self.logger.info(f"浠庢湰鍦扮洰褰曡幏鍙栧埌 {len(sector_dirs)} 涓澘鍧?)
                return sector_dirs
                
            # 灏濊瘯浠巗tock_sectors_mapping妯″潡鑾峰彇
            try:
                # 灏嗗綋鍓嶅伐浣滅洰褰曟坊鍔犲埌sys.path
                sys.path.append(os.getcwd())
                # 灏濊瘯瀵煎叆
                try:
                    from stock_sectors_mapping import get_tng_sectors
                    sector_list = get_tng_sectors()
                    self.logger.info(f"浠巊et_tng_sectors鑾峰彇 {len(sector_list)} 涓澘鍧?)
                    return sector_list
                except ImportError:
                    # 濡傛灉瀵煎叆澶辫触锛屽皾璇曚粠to鐩綍瀵煎叆
                    try:
                        sys.path.append(str(get_project_root()))
                        from to.stock_sectors_mapping import get_tng_sectors
                        sector_list = get_tng_sectors()
                        self.logger.info(f"浠巘o.stock_sectors_mapping鑾峰彇 {len(sector_list)} 涓澘鍧?)
                        return sector_list
                    except ImportError:
                        self.logger.warning("鏃犳硶瀵煎叆get_tng_sectors鍑芥暟")
            except Exception as e:
                self.logger.error(f"灏濊瘯鑾峰彇鏉垮潡鍒楄〃鏃跺嚭閿? {e}")
                
            # 濡傛灉閮藉け璐ヤ簡锛岃繑鍥炵┖鍒楄〃
            return []
            
        except Exception as e:
            self.logger.error(f"鑾峰彇鎵€鏈夋澘鍧楀垪琛ㄦ椂鍑洪敊: {e}")
            return []

    def _detect_single_stock_change_points(self, stock_code, stock_df, indicators=None):
        """妫€娴嬪崟涓偂绁ㄧ殑鍙樺寲鐐?
        
        Args:
            stock_code: 鑲＄エ浠ｇ爜
            stock_df: 鑲＄エ鏁版嵁DataFrame
            indicators: 鎸囨爣鍒楄〃(鍙€?锛岄粯璁や粎浣跨敤future_return_2d
        
        Returns:
            dict: 鍙樺寲鐐规娴嬬粨鏋?
        """
        from ruptures import Pelt, Binseg, Window
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        self.logger.info(f"寮€濮嬫娴嬭偂绁?{stock_code} 鐨勫彉鍖栫偣...")
        
        try:
            # 纭繚鎸夋棩鏈熸帓搴?
            stock_df = stock_df.sort_values('date_str')
            
            # 妫€鏌ユ槸鍚︽湁瓒冲鐨勬暟鎹?
            if len(stock_df) < 20:  # 闇€瑕佽嚦灏?0涓暟鎹偣
                self.logger.warning(f"鑲＄エ {stock_code} 鏁版嵁鐐逛笉瓒?({len(stock_df)}涓?锛岃烦杩囧彉鍖栫偣妫€娴?)
                return None
            
            # 妫€鏌uture_return_2d鏄惁瀛樺湪锛堝繀椤绘寚鏍囷級
            if 'future_return_2d' not in stock_df.columns:
                self.logger.warning(f"鑲＄エ {stock_code} 缂哄皯蹇呰鎸囨爣future_return_2d锛岃烦杩囧彉鍖栫偣妫€娴?)
                return None
            
            # 淇濆瓨鏃ユ湡搴忓垪鐢ㄤ簬缁撴灉璁板綍
            dates = stock_df['date_str'].values
            
            # 鑾峰彇future_return_2d搴忓垪
            future_returns = stock_df['future_return_2d'].values
            
            # 纭繚future_returns鏄湁鏁堢殑鏁板€兼暟缁?
            if not isinstance(future_returns, np.ndarray):
                self.logger.warning(f"鑲＄エ {stock_code} 鐨刦uture_return_2d涓嶆槸numpy鏁扮粍锛屽皾璇曡浆鎹?)
                try:
                    future_returns = np.array(future_returns, dtype=float)
                except Exception as e:
                    self.logger.error(f"杞崲future_return_2d鍒皀umpy鏁扮粍澶辫触: {e}")
                    return None
            
            # 澶勭悊缂哄け鍊?
            try:
                # 妫€鏌ユ槸鍚︽湁NaN鍊?
                nan_mask = np.isnan(future_returns)
                if np.any(nan_mask):
                    # 浣跨敤鍧囧€煎～鍏呯己澶卞€?
                    valid_returns = future_returns[~nan_mask]
                    if len(valid_returns) == 0:
                        self.logger.warning(f"鑲＄エ {stock_code} 鐨刦uture_return_2d鍏ㄤ负NaN锛岃烦杩囧彉鍖栫偣妫€娴?)
                        return None
                    
                    mean_return = np.mean(valid_returns)
                    future_returns = np.nan_to_num(future_returns, nan=mean_return)
            except Exception as e:
                self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鐨刦uture_return_2d鏃跺嚭閿? {e}")
                # 灏濊瘯涓€绉嶆洿瀹夊叏鐨勬柟寮忓鐞嗘暟鎹?
                try:
                    # 鎵嬪姩澶勭悊NaN鍜屾棤鏁堝€?
                    clean_returns = []
                    for val in future_returns:
                        try:
                            fval = float(val)
                            if np.isfinite(fval):
                                clean_returns.append(fval)
                            else:
                                clean_returns.append(0.0)  # 鐢?鏇挎崲鏃犳晥鍊?
                        except (ValueError, TypeError):
                            clean_returns.append(0.0)
                    
                    if not clean_returns:
                        return None
                    
                    future_returns = np.array(clean_returns, dtype=float)
                except Exception as e2:
                    self.logger.error(f"灏濊瘯淇future_returns澶辫触: {e2}")
                    return None
            
            # 杈撳嚭future_return_2d缁熻淇℃伅
            returns_mean = np.mean(future_returns)
            returns_std = np.std(future_returns)
            returns_min = np.min(future_returns)
            returns_max = np.max(future_returns)
            self.logger.info(f"鑲＄エ {stock_code} future_return_2d缁熻: 鍧囧€?{returns_mean:.4f}, 鏍囧噯宸?{returns_std:.4f}, 鏈€灏忓€?{returns_min:.4f}, 鏈€澶у€?{returns_max:.4f}")
            
            # 灏嗘暟鎹浆鎹负浜岀淮鏁扮粍锛堜负浜嗗吋瀹箁uptures搴撶殑杈撳叆鏍煎紡锛?
            indicator_matrix = future_returns.reshape(-1, 1)
            
            # 鏍囧噯鍖栨暟鎹?
            scaler = StandardScaler()
            indicator_matrix_scaled = scaler.fit_transform(indicator_matrix)
            
            # 璁＄畻淇″彿闀垮害鍜屾渶灏忓垎娈甸暱搴?
            n_samples = len(indicator_matrix_scaled)
            
            # 鏍规嵁鏁版嵁闀垮害鍔ㄦ€佽皟鏁村弬鏁?
            if n_samples > 100:
                min_size = min(10, n_samples // 20)  # 鍑忓皬鏈€灏忓垎娈甸暱搴︼紝鏈€灏忎负10鎴栨暟鎹暱搴︾殑5%
                max_points = min(20, n_samples // 10)  # 鏈€澶氭娴嬬偣鏁颁笉瓒呰繃20鎴栨暟鎹暱搴︾殑10%
            else:
                min_size = min(5, n_samples // 10)  # 瀵逛簬鐭簭鍒楋紝浣跨敤鏇村皬鐨勫垎娈甸暱搴?
                max_points = min(10, n_samples // 5)  # 瀵逛簬鐭簭鍒楋紝闄愬埗鍙樺寲鐐规暟閲?
            
            # 闄嶄綆鎯╃綒鍊?- 浣跨敤鑷€傚簲鎯╃綒鍊?
            standard_penalty = 0.5 * np.log(n_samples)  # 澶у箙闄嶄綆鎯╃綒锛屼娇绠楁硶鏇存晱鎰?
            
            # 鍙樺寲鐐规娴嬬粨鏋滃鍣?
            change_points_results = []
            
            # === 鏂规硶1: Pelt绠楁硶 ===
            try:
                pelt_model = Pelt(model="l2", min_size=min_size)
                pelt_points = pelt_model.fit_predict(indicator_matrix_scaled, pen=standard_penalty)
                # 杩囨护鎺夊簭鍒楁湯灏剧偣
                pelt_points = [p for p in pelt_points if p < n_samples]
                
                if len(pelt_points) > 0:
                    self.logger.info(f"Pelt-L2 妫€娴嬪埌 {len(pelt_points)} 涓彉鍖栫偣")
                    change_points_results.append(('pelt_l2', pelt_points))
                
                # 濡傛灉妫€娴嬩笉鍒扮偣锛岃繘涓€姝ラ檷浣庢儵缃氬€?
                if len(pelt_points) < 2:
                    lower_penalty = standard_penalty * 0.5
                    pelt_points = pelt_model.fit_predict(indicator_matrix_scaled, pen=lower_penalty)
                    pelt_points = [p for p in pelt_points if p < n_samples]
                    
                    if len(pelt_points) > 0:
                        self.logger.info(f"Pelt-L2(浣庢儵缃? 妫€娴嬪埌 {len(pelt_points)} 涓彉鍖栫偣")
                        change_points_results.append(('pelt_l2_low', pelt_points))
            except Exception as e:
                self.logger.warning(f"Pelt-L2 绠楁硶澶辫触: {e}")
            
            # === 鏂规硶2: Binseg绠楁硶 ===
            try:
                # 浜屽垎鍓插垎鍓叉硶锛屼笉闇€瑕佹儵缃氾紝鐩存帴鎸囧畾鍒嗗壊鏁伴噺
                binseg_model = Binseg(model="l2", min_size=min_size)
                binseg_points = binseg_model.fit_predict(indicator_matrix_scaled, n_bkps=max_points)
                binseg_points = [p for p in binseg_points if p < n_samples]
                
                if len(binseg_points) > 0:
                    self.logger.info(f"Binseg 妫€娴嬪埌 {len(binseg_points)} 涓彉鍖栫偣")
                    change_points_results.append(('binseg', binseg_points))
            except Exception as e:
                self.logger.warning(f"Binseg 绠楁硶澶辫触: {e}")
            
            # === 鏂规硶3: Window绠楁硶 ===
            try:
                # 绐楀彛瀹藉害涓烘暟鎹暱搴︾殑10%锛屾渶灏?0涓偣
                window_width = max(10, n_samples // 10)
                window_model = Window(width=window_width, model="l2", min_size=min_size)
                window_points = window_model.fit_predict(indicator_matrix_scaled, n_bkps=max_points)
                window_points = [p for p in window_points if p < n_samples]
                
                if len(window_points) > 0:
                    self.logger.info(f"Window 妫€娴嬪埌 {len(window_points)} 涓彉鍖栫偣")
                    change_points_results.append(('window', window_points))
            except Exception as e:
                self.logger.warning(f"Window 绠楁硶澶辫触: {e}")
            
            # 閫夋嫨鍙樺寲鐐圭粨鏋?
            if not change_points_results:
                self.logger.warning(f"鑲＄エ {stock_code} 鏈兘浣跨敤浠讳綍鏂规硶妫€娴嬪埌鍙樺寲鐐?)
                return None
            
            # 璇勪及姣忕绠楁硶缁撴灉鐨勮川閲?
            best_algorithm = None
            final_change_points = []
            
            # 鍋忓ソ鍩轰簬鏈€澶ф敹鐩婄巼鍙樺寲鐨勬娴嬬偣
            best_score = -1
            
            for alg_name, points in change_points_results:
                if len(points) < 2:  # 鑷冲皯闇€瑕佷袱涓彉鍖栫偣
                    continue
                
                # 璁＄畻鍙樺寲鐐瑰墠鍚庣殑鏀剁泭鐜囧彉鍖栧箙搴?
                return_changes = []
                for i in range(len(points)-1):
                    start, end = points[i], points[i+1]
                    if start < len(future_returns) and end < len(future_returns):
                        segment1 = future_returns[start:end]
                        mean1 = np.mean(segment1)
                        
                        # 璁＄畻鐩搁偦娈电殑鍧囧€煎彉鍖?
                        if i+2 < len(points):
                            next_end = points[i+2]
                            if next_end < len(future_returns):
                                segment2 = future_returns[end:next_end]
                                mean2 = np.mean(segment2)
                                return_changes.append(abs(mean2 - mean1))
                
                # 濡傛灉鏈夎冻澶熺殑鍙樺寲锛岃绠楀钩鍧囧彉鍖栧箙搴?
                if return_changes:
                    avg_change = np.mean(return_changes)
                    # 鍙樺寲骞呭害瓒婂ぇ瓒婂ソ锛屽悓鏃惰€冭檻鍙樺寲鐐规暟閲忛€備腑
                    score = avg_change * min(1.0, 10.0/len(points))
                    
                    if score > best_score:
                        best_score = score
                        best_algorithm = alg_name
                        final_change_points = points
            
            # 濡傛灉鏈兘閫夊嚭鏈€浣崇畻娉曪紝閫夋嫨妫€娴嬪埌鍙樺寲鐐规渶澶氱殑绠楁硶
            if not best_algorithm:
                sorted_results = sorted(change_points_results, key=lambda x: len(x[1]), reverse=True)
                best_algorithm, final_change_points = sorted_results[0]
            
            # 璁板綍鍙樺寲鐐?
            if final_change_points:
                # 纭繚鍙樺寲鐐规槸鏁存暟
                final_change_points = [int(cp) for cp in final_change_points]
                
                # 鎸夐『搴忔帓搴?
                final_change_points.sort()
                
                # 鑾峰彇鍙樺寲鐐瑰搴旂殑鏃ユ湡
                cp_dates = dates[final_change_points]
                
                # 鑾峰彇鍙樺寲鐐瑰搴旂殑future_return_2d鍊?
                cp_returns = future_returns[final_change_points]
                
                # 鍒涘缓缁撴灉瀛楀吀
                result = {
                    'change_points': final_change_points,
                    'dates': cp_dates,
                    'returns': cp_returns,
                    'algorithm': best_algorithm
                }
                
                self.logger.info(f"鑲＄エ {stock_code} 鏈€缁堜娇鐢?{best_algorithm} 妫€娴嬪埌 {len(final_change_points)} 涓彉鍖栫偣")
                return result
            else:
                self.logger.warning(f"鑲＄エ {stock_code} 鏈兘妫€娴嬪埌浠讳綍鍙樺寲鐐?)
                return None
            
        except Exception as e:
            self.logger.error(f"妫€娴嬭偂绁?{stock_code} 鍙樺寲鐐规椂鍑洪敊: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _analyze_stock_change_point_patterns(self, stock_features_df, change_point_results, relations_df=None):
        """
        鍒嗘瀽涓偂鍙樺寲鐐圭殑鐗瑰緛妯″紡锛屼娇鐢ㄩ噺鍖栭噾铻嶇殑Regime Detection鎶€鏈?
        Args:
            stock_features_df: 涓偂鐗瑰緛DataFrame
            change_point_results: 鍙樺寲鐐规娴嬬粨鏋?
            relations_df: 鑲＄エ鏉垮潡鍏宠仈缁撴灉DataFrame锛屽彲閫夊弬鏁?
            
        Returns:
            dict: 鐗瑰緛妯″紡
        """
        import multiprocessing as mp
        from functools import partial
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        import hashlib
        import pickle
        import os
        
        # 鍒涘缓缂撳瓨鐩綍
        cache_dir = os.path.join(self.model_dir, "pattern_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 鐢熸垚缂撳瓨閿?- 鍩轰簬杈撳叆鏁版嵁鐨勫搱甯屽€?
        cache_key = ""
        try:
            # 浣跨敤杈撳叆鏁版嵁鐨勫叧閿壒寰佸垱寤哄敮涓€鍝堝笇
            features_hash = hashlib.md5(str(stock_features_df.shape).encode()).hexdigest()
            change_points_hash = hashlib.md5(str(len(change_point_results)).encode()).hexdigest()
            relations_hash = "" if relations_df is None else hashlib.md5(str(relations_df.shape).encode()).hexdigest()
            cache_key = f"{features_hash}_{change_points_hash}_{relations_hash}"
            cache_file = os.path.join(cache_dir, f"pattern_{cache_key}.pkl")
            
            # 妫€鏌ョ紦瀛樻槸鍚﹀瓨鍦?
            if os.path.exists(cache_file):
                self.logger.info(f"浠庣紦瀛樺姞杞藉彉鍖栫偣妯″紡鍒嗘瀽缁撴灉: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"缂撳瓨妫€鏌ュけ璐? {e}")
        
        # 鍘熷鏂规硶鐨勪唬鐮佺户缁墽琛?..
        self.logger.info("寮€濮嬪垎鏋愪釜鑲″彉鍖栫偣鐗瑰緛妯″紡...")
        
        # 瀹氫箟鍙樺寲鐐规ā寮忕壒寰佺粨鏋?
        patterns_collection = {}
        
        # 瀹氫箟鍏抽敭鐗瑰緛鍒楄〃 - 鎸夐噸瑕佹€ф帓搴?
        key_price_features = [
            'close', 'open', 'high', 'low',              # 鍩虹浠锋牸鐗瑰緛
            'change_percent', 'amplitude',               # 鍙樺姩骞呭害鐗瑰緛
            'volume', 'turnover_rate',                   # 鎴愪氦閲忕壒寰?
        ]
        
        key_technical_features = [
            'ma5_10_ratio', 'ma5_20_ratio', 'ma10_20_ratio',  # 鍧囩嚎鍏崇郴鐗瑰緛
            'volatility_5d', 'volatility_10d', 'volatility_20d',  # 娉㈠姩鐜囩壒寰?
            'momentum_5d', 'momentum_10d', 'momentum_20d',  # 鍔ㄩ噺鐗瑰緛
            'volume_ratio_5', 'volume_ratio_10',  # 鐩稿鎴愪氦閲忕壒寰?
            'rsi6', 'rsi14',  # RSI鎸囨爣
            'macd_line', 'macd_signal',  # MACD鎸囨爣
            'trend_strength', 'consecutive_up_days', 'consecutive_down_days',  # 瓒嬪娍寮哄害鍜岃繛缁€?
            'support_distance_pct', 'resistance_distance_pct',  # 鏀拺闃诲姏浣?
            'tech_score', 'market_sentiment', 'signal_strength',  # 鎶€鏈瘎鍒?
        ]
        
        # 妫€鏌ユ槸鍚︽彁渚涗簡鑲＄エ鏉垮潡鍏宠仈鏁版嵁
        if relations_df is not None and not relations_df.empty:
            self.logger.info(f"浣跨敤鍗曠嫭鐨勮偂绁ㄦ澘鍧楀叧鑱旀暟鎹?鍖呭惈{len(relations_df)}鏉¤褰?杩涜鍒嗘瀽")
        
        key_sector_features = [
            'relative_return',         # 涓偂鐩稿鏉垮潡鏀剁泭鐜?
            'relative_turnover',       # 涓偂鐩稿鏉垮潡鎹㈡墜鐜?
            'relative_volume',         # 涓偂鐩稿鏉垮潡鎴愪氦閲?
            'relative_momentum',       # 涓偂鐩稿鏉垮潡鍔ㄩ噺
            'sector_deviation',        # 涓偂瀵规澘鍧楃殑鑳岀绋嬪害
            'is_sector_leader',        # 鏄惁涓烘澘鍧楅緳澶?
        ]
        
        # 鏂板锛氬競鍦虹幆澧冪壒寰?
        key_market_features = [
            'market_type', 'market_group', 'market_cycle', 'market_trend_strength',  # 甯傚満鐘舵€?
            'momentum_vs_market_rsi', 'volatility_vs_market', 'rsi_vs_market',  # 鐩稿甯傚満鎸囨爣
            'rsi_diff_market', 'macd_vs_market', 'macd_diff_market',  # 甯傚満宸紓鎸囨爣
            'correlation_with_cyb',  # 涓庡垱涓氭澘鐩稿叧鎬?
        ]
        
        # 鏂板锛氬舰鎬佽瘑鍒壒寰?
        key_pattern_features = [
            'pattern_doji', 'pattern_hammer', 'pattern_inverted_hammer',  # K绾垮舰鎬?
            'pattern_engulfing_bullish', 'pattern_engulfing_bearish',  # 鍚炲櫖褰㈡€?
            'pattern_morning_star', 'pattern_evening_star',  # 鍚槑/榛勬槒涔嬫槦
            'pattern_three_white_soldiers', 'pattern_three_black_crows',  # 涓夌櫧鍏?涓夐粦楦?
            'pattern_hanging_man', 'pattern_shooting_star',  # 鍚婇绾?娴佹槦绾?
            'pattern_head_shoulders', 'pattern_inverse_head_shoulders',  # 澶磋偐椤?搴?
            'pattern_double_top', 'pattern_double_bottom',  # 鍙岄《/鍙屽簳
            'pattern_triangle_ascending', 'pattern_triangle_descending',  # 涓婂崌/涓嬮檷涓夎褰?
            'pattern_triangle_symmetrical', 'pattern_flag_bullish', 'pattern_flag_bearish',  # 瀵圭О涓夎褰€佹棗褰?
            'chart_pattern_strength', 'candlestick_pattern_strength',  # 褰㈡€佸己搴?
            'trend_state', 'trend_state_value',  # 瓒嬪娍鐘舵€?
            'stock_state', 'stock_state_value',  # 鑲＄エ鐘舵€?
        ]
        
        # 鏂板锛氶珮绾ф妧鏈粍鍚堢壒寰?
        key_composite_features = [
            'rsi_macd_bull_signal', 'rsi_macd_bear_signal',  # RSI涓嶮ACD缁勫悎淇″彿
            'overbought_macd_bearish', 'oversold_macd_bullish',  # 瓒呬拱/鍗栧尯鍩熶俊鍙?
            'momentum_volatility_ratio',  # 椋庨櫓璋冩暣鍔ㄩ噺姣?
            'volume_momentum_confirm', 'volume_momentum_divergence',  # 閲忎环鍏崇郴
            'indicator_consensus', 'strong_bull_consensus', 'strong_bear_consensus',  # 鎸囨爣涓€鑷存€?
            'winning_bull_signal', 'winning_bear_signal',  # 鎴樻棤涓嶅厠淇″彿
        ]
        
        # 鏂板锛氭澘鍧楁定璺屽仠缁撴瀯鐗瑰緛
        key_board_structure_features = [
            'limit_up_count', 'limit_down_count', 'near_limit_up_count',  # 娑ㄨ穼鍋滄暟閲?
            'limit_up_ratio', 'limit_down_ratio',  # 娑ㄨ穼鍋滃崰姣?
            'one_word_limit_ratio', 'broken_ratio',  # 涓€瀛楁澘銆佺偢鏉垮崰姣?
            'max_consecutive_limit_up',  # 鏈€楂樿繛鏉挎暟
            'board_heat_index',  # 鐑害鎸囨暟
            'yesterday_limit_up_premium',  # 鏄ㄦ棩娑ㄥ仠婧环
        ]
        
        # 鏂板锛氬競鍦烘潯浠朵笅鐨勭壒寰?
        key_conditional_features = [
            'momentum_5d_in_bull', 'momentum_10d_in_bull', 'momentum_20d_in_bull',  # 鐗涘競鏉′欢涓嬪姩閲?
            'momentum_5d_in_bear', 'momentum_10d_in_bear', 'momentum_20d_in_bear',  # 鐔婂競鏉′欢涓嬪姩閲?
            'momentum_5d_in_extreme', 'momentum_10d_in_extreme', 'momentum_20d_in_extreme',  # 鏋佺甯傚満鏉′欢涓嬪姩閲?
            'momentum_5d_in_neutral', 'momentum_10d_in_neutral', 'momentum_20d_in_neutral',  # 闇囪崱甯傛潯浠朵笅鍔ㄩ噺
            'rsi_bull_signal', 'rsi_bull_overbought', 'rsi_bear_signal', 'rsi_bear_oversold',  # 鐗涚唺甯俁SI淇″彿
            'rsi_extreme_high', 'rsi_extreme_low', 'rsi_extreme_reversal', 'rsi_neutral_middle',  # 鏋佺/涓€SI淇″彿
            'macd_bull_cross', 'macd_bear_cross', 'macd_extreme_deviation',  # MACD鏉′欢淇″彿
            'macd_extreme_divergence', 'macd_neutral_zero_cross',  # MACD鏉′欢淇″彿
            'volume_bull_signal', 'volume_bear_signal',  # 鎴愪氦閲忔潯浠朵俊鍙?
            'volume_extreme_high', 'volume_extreme_low', 'volume_neutral_signal',  # 鎴愪氦閲忔潯浠朵俊鍙?
            'bull_momentum_volume', 'bear_oversold_bounce',  # 澶嶅悎鏉′欢淇″彿
            'extreme_oversold_high_vol', 'extreme_overbought_high_vol',  # 鏋佺甯傚満淇″彿
            'extreme_volume_price_divergence', 'neutral_volatility_signal',  # 鏋佺/涓€у競鍦轰俊鍙?
        ]
        
        # 鍚堝苟鎵€鏈夊叧閿壒寰?
        all_key_features = (key_price_features + key_technical_features + key_sector_features + 
                           key_market_features + key_pattern_features + key_composite_features + 
                           key_board_structure_features + key_conditional_features)
        
        # 鍒涘缓杩斿洖鍊煎瓧鍏?
        regime_patterns = {}
        
        # 鍒涘缓鍏变韩缁撴灉鍒楄〃
        manager = mp.Manager()
        positive_regime_changes = manager.list()
        negative_regime_changes = manager.list()
        patterns_collection_shared = manager.dict()
        
        # 澶勭悊鍗曚釜鑲＄エ鐨勫嚱鏁?
        def process_single_stock(stock_code, result, all_key_features):
            """澶勭悊鍗曞彧鑲＄エ鐨勫彉鍖栫偣鍒嗘瀽"""
            try:
                stock_data = result.get(stock_code)
                if not stock_data or 'change_points' not in stock_data:
                    return {}
                
                # 鎻愬彇璇ヨ偂绁ㄧ殑鍙樺寲鐐瑰拰瀵瑰簲鐨勬澘鍧椾俊鎭?
                change_points = stock_data['change_points']
                if not change_points:
                    return {}
                
                positive_regime_changes = []
                negative_regime_changes = []
                
                # 閬嶅巻姣忎釜鍙樺寲鐐癸紝杩涜鍒嗘瀽
                for cp_idx, cp_data in enumerate(change_points):
                    pattern = {}
                    for feature in all_key_features:
                        if feature in cp_data:
                            pattern[feature] = cp_data[feature]
                    
                    # 娣诲姞鑲＄エ浠ｇ爜鍜屽彉鍖栫偣绱㈠紩
                    pattern['stock_code'] = stock_code
                    pattern['cp_idx'] = cp_idx
                    
                    # 鑾峰彇鏈潵鏀剁泭鐜?
                    cp_return = None
                    if 'future_return_2d' in cp_data:
                        cp_return = cp_data['future_return_2d']
                    
                    # 鏍规嵁鏈潵鏀剁泭灏嗘ā寮忓垎绫讳负姝ｅ悜鎴栬礋鍚戯紝娣诲姞鍒板叡浜垪琛?
                    if cp_return is not None:
                        try:
                            cp_return = float(cp_return)
                            if not np.isnan(cp_return) and np.isfinite(cp_return):
                                if cp_return > 0.02:  # 2%浠ヤ笂瑙嗕负鏄捐憲姝ｅ悜
                                    positive_regime_changes.append(pattern)
                                elif cp_return < -0.02:  # -2%浠ヤ笅瑙嗕负鏄捐憲璐熷悜
                                    negative_regime_changes.append(pattern)
                        except (ValueError, TypeError):
                            continue
                
                return {
                    'positive_patterns': positive_regime_changes,
                    'negative_patterns': negative_regime_changes
                }
            except Exception as e:
                self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鏃跺嚭閿? {e}")
                return {}
        
        # 浣跨敤澶氱嚎绋嬪鐞嗚偂绁紙閬垮厤澶氳繘绋嬬殑搴忓垪鍖栭棶棰橈級
        # 纭畾鍚堥€傜殑绾跨▼鏁伴噺
        num_cores = max(1, min(mp.cpu_count() - 1, len(change_point_results)))
        self.logger.info(f"浣跨敤 {num_cores} 涓嚎绋嬪垎鏋?{len(change_point_results)} 鍙偂绁ㄧ殑鍙樺寲鐐?..")
        
        # 鍑嗗浠诲姟鍒楄〃
        tasks = list(change_point_results.items())
        
        # 浣跨敤tqdm鏄剧ず杩涘害
        total_processed = 0
        valid_patterns = 0
        
        # 鎵归噺澶勭悊锛屾瘡鎵规鏈€澶氬鐞?0涓偂绁?
        batch_size = min(50, len(tasks))
        
        # 浣跨敤绾跨▼姹犲鐞?
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                
                # 鎻愪氦鎵瑰鐞嗕换鍔?
                futures = []
                for code, result in batch:
                    futures.append(executor.submit(process_single_stock, code, result, all_key_features))
                
                # 澶勭悊缁撴灉
                for future in tqdm(futures, desc=f"鍒嗘瀽鍙樺寲鐐圭壒寰?鎵规 {i//batch_size+1}/{(len(tasks)+batch_size-1)//batch_size}"):
                    result = future.result()
                    if result:
                        stock_code = code
                        patterns_collection_shared[stock_code] = result
                        valid_patterns += 1
                    total_processed += 1
        
        # 杞崲鍏变韩鏁版嵁缁撴瀯涓烘櫘閫歅ython鏁版嵁缁撴瀯
        patterns_collection = dict(patterns_collection_shared)
        positive_regime_changes = list(positive_regime_changes)
        negative_regime_changes = list(negative_regime_changes)
        
        # 瀵规墍鏈夋敹闆嗗埌鐨勬渶浣虫ā寮忚繘琛岃仛绫诲垎鏋?
        if valid_patterns > 0:
            self.logger.info(f"鎴愬姛浠?{valid_patterns}/{total_processed} 鍙偂绁ㄤ腑鎻愬彇鍒版湁鏁堟ā寮?)
            
            # 灏嗘敹闆嗗埌鐨勬ā寮忚浆鎹负鐗瑰緛鍚戦噺杩涜鑱氱被鍒嗘瀽
            patterns_features = []
            pattern_keys = []
            
            # 閫夋嫨鐢ㄤ簬鑱氱被鐨勫叧閿壒寰?- 鍏呭垎鍒╃敤鎴戜滑璁＄畻鐨勯珮绾х壒寰?
            clustering_features = [
                'future_return_2d',                # 鏈潵鏀剁泭鐜?
                'change_percent',                  # 褰撴棩娑ㄨ穼骞?
                'momentum_10d',                    # 涓湡鍔ㄩ噺
                'volatility_20d',                  # 娉㈠姩鐜?
                'volume_ratio_5',                  # 鎴愪氦閲忔瘮
                'turnover_rate',                   # 鎹㈡墜鐜?
                'relative_return',                 # 鐩稿鏉垮潡鏀剁泭
                'tech_combo_state',                # 鎶€鏈寚鏍囩粍鍚堢姸鎬?
                'price_momentum_state',            # 浠锋牸鍔ㄨ兘鐘舵€?
                'volume_state',                    # 鎴愪氦閲忕姸鎬?
                'adjusted_tech_state',             # 璋冩暣鍚庣殑鎶€鏈姸鎬?
                'sector_rotation_effect',          # 鏉垮潡杞姩鏁堝簲
                'board_heat_index',                # 鏉垮潡鐑害鎸囨暟
                'limit_up_ratio',                  # 娑ㄥ仠鍗犳瘮
                'rsi14',                           # RSI鎸囨爣
                'market_group',                    # 甯傚満鐜鍒嗙粍
                'indicator_consensus',             # 鎸囨爣涓€鑷存€?
                'regime_price_trend',              # 鍖洪棿浠锋牸瓒嬪娍
                'regime_relative_performance',     # 鍖洪棿鐩稿琛ㄧ幇
            ]
            
            # 鎻愬彇鐗瑰緛鍚戦噺
            for stock_code, pattern in patterns_collection.items():
                feature_vector = []
                for feature in clustering_features:
                    if feature in pattern:
                        feature_vector.append(pattern[feature])
                    else:
                        feature_vector.append(0)  # 缂哄け鍊肩敤0濉厖
                
                if len(feature_vector) == len(clustering_features):  # 纭繚鐗瑰緛瀹屾暣
                    patterns_features.append(feature_vector)
                    pattern_keys.append(stock_code)
            
            # 鍑嗗鑱氱被鍒嗘瀽
            try:
                # 濡傛灉鏈夎冻澶熺殑鏍锋湰杩涜鑱氱被
                if len(patterns_features) >= 5:
                    # 鏍囧噯鍖栫壒寰?
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(patterns_features)
                    
                    # 闄嶇淮浠ヤ究鍙鍖?
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    reduced_features = pca.fit_transform(scaled_features)
                    
                    # 鑱氱被鍒嗘瀽
                    from sklearn.cluster import KMeans
                    # 浣跨敤鑲橀儴娉曞垯纭畾鏈€浣宠仛绫绘暟
                    optimal_clusters = min(5, len(patterns_features) // 5 + 1)  # 鑷冲皯1涓仛绫伙紝鏈€澶?涓?
                    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(scaled_features)
                    
                    # 涓烘瘡涓ā寮忔坊鍔犺仛绫绘爣绛?
                    for i, stock_code in enumerate(pattern_keys):
                        if i < len(cluster_labels):
                            patterns_collection[stock_code]['cluster'] = int(cluster_labels[i])
                            patterns_collection[stock_code]['pca_x'] = float(reduced_features[i, 0])
                            patterns_collection[stock_code]['pca_y'] = float(reduced_features[i, 1])
                    
                    # 鍒嗘瀽姣忎釜鑱氱被鐨勭壒寰?
                    cluster_info = {}
                    for cluster_id in range(optimal_clusters):
                        # 鑾峰彇璇ヨ仛绫荤殑鎵€鏈夋牱鏈?
                        cluster_samples = [patterns_features[i] for i in range(len(patterns_features)) 
                                        if cluster_labels[i] == cluster_id]
                        if cluster_samples:
                            # 璁＄畻鑱氱被涓績
                            cluster_center = np.mean(cluster_samples, axis=0)
                            # 璁＄畻璇ヨ仛绫荤殑骞冲潎鏈潵鏀剁泭
                            future_returns = [sample[clustering_features.index('future_return_2d')] 
                                            for sample in cluster_samples]
                            avg_return = np.mean(future_returns)
                            
                            # 淇濆瓨鑱氱被淇℃伅
                            cluster_info[cluster_id] = {
                                'size': len(cluster_samples),
                                'avg_return': float(avg_return),
                                'center': [float(x) for x in cluster_center],
                                'feature_importance': {}
                            }
                            
                            # 璁＄畻鐗瑰緛閲嶈鎬?
                            for i, feature in enumerate(clustering_features):
                                if i < len(cluster_center):
                                    cluster_info[cluster_id]['feature_importance'][feature] = abs(float(cluster_center[i]))
                
                    # 娣诲姞鑱氱被鍒嗘瀽缁撴灉
                    regime_patterns['cluster_analysis'] = {
                        'num_clusters': optimal_clusters,
                        'clusters': cluster_info,
                        'feature_names': clustering_features
                    }
            except Exception as e:
                self.logger.warning(f"鑱氱被鍒嗘瀽鏃跺嚭閿? {e}")
            
            # 鍑嗗鑱氱被缁撴灉
            regime_patterns = {
                'positive_patterns': positive_regime_changes,
                'negative_patterns': negative_regime_changes,
                'stock_best_patterns': patterns_collection,
                'key_features': all_key_features,
                'clustering_features': clustering_features,
                'pattern_keys': pattern_keys
            }
            
            # 璁＄畻鍚勭壒寰佸鏈潵鏀剁泭鐜囩殑鐩稿叧鎬?
            feature_correlations = {}
            combined_patterns = positive_regime_changes + negative_regime_changes
            if combined_patterns:
                # 娣诲姞瀹夊叏妫€鏌?- 纭繚鎵€鏈夋ā寮忎腑鐨勭壒寰佸€奸兘鏄暟鍊肩被鍨?
                valid_combined_patterns = []
                for pattern in combined_patterns:
                    if 'future_return_2d' in pattern and pattern['future_return_2d'] is not None:
                        try:
                            # 纭繚鏈潵鏀剁泭鏄暟鍊肩被鍨?
                            pattern['future_return_2d'] = float(pattern['future_return_2d'])
                            valid_combined_patterns.append(pattern)
                        except (ValueError, TypeError):
                            continue
                    
                if not valid_combined_patterns:
                    self.logger.warning("娌℃湁鍖呭惈鏈夋晥future_return_2d鐨勬ā寮忥紝璺宠繃鐗瑰緛鐩稿叧鎬у垎鏋?)
                else:
                    combined_patterns = valid_combined_patterns
                    
                for feature in all_key_features:
                    # 鏀堕泦鐗瑰緛鍊煎拰瀵瑰簲鐨勬湭鏉ユ敹鐩婄巼
                    feature_values = []
                    return_values = []
                    for pattern in combined_patterns:
                        if feature in pattern and 'future_return_2d' in pattern:
                            feature_val = pattern[feature]
                            return_val = pattern['future_return_2d']
                            # 纭繚鍊兼槸鏁板€肩被鍨嬩笖涓嶆槸None
                            if (feature_val is not None and return_val is not None):
                                try:
                                    # 灏濊瘯杞崲涓篺loat
                                    feature_val = float(feature_val)
                                    return_val = float(return_val)
                                    # 妫€鏌ユ槸鍚︿负鏈夐檺鍊?
                                    if np.isfinite(feature_val) and np.isfinite(return_val):
                                        feature_values.append(feature_val)
                                        return_values.append(return_val)
                                except (ValueError, TypeError):
                                    continue
                    
                    # 璁＄畻鐩稿叧鎬?
                    if len(feature_values) >= 10:  # 鑷冲皯闇€瑕?0涓暟鎹偣
                        try:
                            # 纭繚杞崲涓簄umpy鏁扮粍骞朵娇鐢╢loat绫诲瀷
                            feature_values_np = np.array(feature_values, dtype=float)
                            return_values_np = np.array(return_values, dtype=float)
                            
                            # 妫€鏌ユ暟缁勬槸鍚﹀寘鍚湁鏁堝€?
                            if (len(feature_values_np) > 0 and len(return_values_np) > 0 and 
                                np.all(np.isfinite(feature_values_np)) and np.all(np.isfinite(return_values_np))):
                                corr = np.corrcoef(feature_values_np, return_values_np)[0, 1]
                                if np.isfinite(corr):  # 妫€鏌ョ浉鍏崇郴鏁版槸鍚︿负鏈夐檺鍊?
                                    feature_correlations[feature] = float(corr)
                        except Exception as e:
                            self.logger.debug(f"璁＄畻鐗瑰緛 {feature} 鐩稿叧鎬ф椂鍑洪敊: {e}")
                
                # 鎸夌浉鍏虫€х粷瀵瑰€兼帓搴忥紝鎵惧嚭鏈€鍏烽娴嬫€х殑鐗瑰緛
                sorted_correlations = sorted(feature_correlations.items(), 
                                           key=lambda x: abs(x[1]), 
                                           reverse=True)
                
                # 淇濆瓨鐗瑰緛鐩稿叧鎬у垎鏋愮粨鏋?
                regime_patterns['feature_correlations'] = dict(sorted_correlations)
                regime_patterns['top_predictive_features'] = dict(sorted_correlations[:20])  # 鍓?0涓渶鍏烽娴嬫€х殑鐗瑰緛
            
            self.logger.info(f"鍒嗘瀽瀹屾垚: 姝ｅ悜妯″紡 {len(positive_regime_changes)}, 璐熷悜妯″紡 {len(negative_regime_changes)}, 鑲＄エ鏈€浣虫ā寮?{len(patterns_collection)}")
        else:
            self.logger.warning("鏈兘浠庝换浣曡偂绁ㄤ腑鎻愬彇鍒版湁鏁堟ā寮?)
            regime_patterns = {
                'positive_patterns': [],
                'negative_patterns': [],
                'stock_best_patterns': {},
                'key_features': all_key_features
            }
        
        # 鍦ㄦ柟娉曟湯灏俱€乺eturn鍓嶆坊鍔犱繚瀛樼紦瀛樼殑浠ｇ爜
        # (姝ゅ娣诲姞浠ヤ笅浠ｇ爜鍦╮eturn regime_patterns鍓?
        try:
            if cache_key:
                self.logger.info(f"淇濆瓨鍙樺寲鐐规ā寮忓垎鏋愮粨鏋滃埌缂撳瓨: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(regime_patterns, f)
        except Exception as e:
            self.logger.warning(f"淇濆瓨缂撳瓨鏂囦欢澶辫触: {e}")
            
        return regime_patterns



    def _predict_stocks_with_change_points(self, predict_features, feature_patterns, relations_df=None):
        """浣跨敤鍙樺寲鐐圭殑鐗瑰緛妯″紡棰勬祴鑲＄エ琛ㄧ幇
        Args:
            predict_features: 棰勬祴鏈熺壒寰佹暟鎹?
            feature_patterns: 鍘嗗彶鐗瑰緛妯″紡鏁版嵁
            relations_df: 鑲＄エ鏉垮潡鍏宠仈缁撴灉DataFrame锛屽彲閫夊弬鏁?
            
        Returns:
            dict: 棰勬祴缁撴灉锛岃偂绁ㄤ唬鐮佷负閿紝棰勬祴淇℃伅涓哄€?
        """
        # 浣跨敤澶氱嚎绋嬬増鏈?
        from to.multithreaded_predict import predict_stocks_with_change_points_mt
        self.logger.info("浣跨敤澶氱嚎绋嬬増鏈娴嬶紝骞惰澶勭悊澶氫釜浜ゆ槗鏃ユ暟鎹?..")
        return predict_stocks_with_change_points_mt(self, predict_features, feature_patterns, relations_df)
    
    def _calculate_weighted_similarity(self, vec1_dict, vec2_dict, feature_weights=None):
        """璁＄畻涓や釜鐗瑰緛瀛楀吀鐨勫姞鏉冪浉浼煎害
        
        Args:
            vec1_dict: 绗竴涓壒寰佸瓧鍏?
            vec2_dict: 绗簩涓壒寰佸瓧鍏?
            feature_weights: 鐗瑰緛鏉冮噸瀛楀吀锛屽鏋滀负None鍒欎娇鐢ㄨ嚜鍔ㄨ绠楃殑鏉冮噸
            
        Returns:
            tuple: (鐩镐技搴﹀垎鏁? 鍖归厤鐨勭壒寰佸垪琛? 鐗瑰緛璐＄尞搴﹀瓧鍏?
        """
        # 鎵惧嚭涓や釜瀛楀吀涓叡鏈夌殑鐗瑰緛
        common_features = []
        
        # 濡傛灉娌℃湁鎻愪緵鏉冮噸锛屽垯涓烘墍鏈夊叡鍚岀壒寰佽祴浜堢浉鍚屾潈閲?
        if feature_weights is None:
            feature_weights = {}
            for feature in set(vec1_dict.keys()).intersection(set(vec2_dict.keys())):
                feature_weights[feature] = 1.0
        
        for feature in feature_weights.keys():
            if feature in vec1_dict and feature in vec2_dict:
                common_features.append(feature)
        
        if not common_features:
            return 0, [], {}
        
        # 鎻愬彇鍏辨湁鐗瑰緛鐨勫€煎拰鏉冮噸
        vec1 = []
        vec2 = []
        weights = []
        
        for feature in common_features:
            vec1.append(vec1_dict[feature])
            vec2.append(vec2_dict[feature])
            weights.append(feature_weights[feature])
        
        # 杞崲涓簄umpy鏁扮粍
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        weights = np.array(weights)
        
        # 鏍囧噯鍖栨潈閲?
        weights = weights / np.sum(weights)
        
        # 璁＄畻鍔犳潈浣欏鸡鐩镐技搴?
        numerator = np.sum(weights * vec1 * vec2)
        denominator = np.sqrt(np.sum(weights * vec1 * vec1)) * np.sqrt(np.sum(weights * vec2 * vec2))
        
        if denominator == 0:
            return 0, common_features, {}
        
        similarity = numerator / denominator
        
        # 灏嗙浉浼煎害闄愬埗鍦?鍒?涔嬮棿
        similarity = max(0, min(1, similarity))
        
        # 璁＄畻姣忎釜鐗瑰緛瀵圭浉浼煎害鐨勮础鐚?
        feature_contributions = {}
        if similarity > 0 and denominator > 0:  # 纭繚鍒嗘瘝澶т簬0
            for i, feature in enumerate(common_features):
                try:
                    # 瀹夊叏璁＄畻鐗瑰緛璐＄尞搴?
                    contribution = weights[i] * vec1[i] * vec2[i] / (similarity * denominator)
                    feature_contributions[feature] = contribution
                except (ZeroDivisionError, RuntimeWarning):
                    # 澶勭悊鍙兘鐨勯櫎闆堕敊璇?
                    feature_contributions[feature] = 0.0
            
            # 鎸夎础鐚害鎺掑簭
            feature_contributions = {k: v for k, v in sorted(
                feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True
            )}
        
        return similarity, common_features, feature_contributions
    
    def _rank_and_select_stocks(self, predictions, top_n=50):
        """鏍规嵁棰勬祴缁撴灉瀵硅偂绁ㄦ帓搴忓苟閫夋嫨姣忎釜浜ゆ槗鏃ユ渶浣崇殑鑲＄エ
        
        Args:
            predictions: 棰勬祴缁撴灉
            top_n: 姣忔棩杩斿洖鐨勮偂绁ㄦ暟閲?
            
        Returns:
            dict: 閫夎偂缁撴灉
        """
        self.logger.info(f"寮€濮嬫寜鏃ユ湡鎺掑簭骞堕€夋嫨姣忓ぉ鍓?{top_n} 鍙偂绁?..")
        
        if not predictions:
            self.logger.warning("娌℃湁鏈夋晥鐨勯娴嬬粨鏋?)
            return {'selected_stocks_by_date': {}, 'dates': [], 'total_stocks': 0, 'date': datetime.now().strftime('%Y%m%d')}
        
        # 鎸夋棩鏈熷垎缁?
        stocks_by_date = {}
        for stock_code, stock_data in predictions.items():
            date = stock_data.get('date')
            if not date:
                continue
            
            if date not in stocks_by_date:
                stocks_by_date[date] = []
            stocks_by_date[date].append(stock_data)
        
        # 瀵规瘡涓棩鏈熺粍鍐呯殑鑲＄エ璁＄畻寰楀垎骞舵帓搴?
        selected_stocks_by_date = {}
        total_selected = 0
        
        for date, stocks in stocks_by_date.items():
            # 璁＄畻缁煎悎寰楀垎:缁撳悎棰勬湡鏀剁泭鐜囥€佹鐜囧拰缃俊搴?
            for stock in stocks:
                # 缁煎悎鍒嗘暟: 棰勬湡鏀剁泭 * 姒傜巼 * 缃俊搴?
                stock['score'] = stock['expected_return'] * stock['probability'] * stock['confidence']
            
            # 鎸夊緱鍒嗘帓搴?
            ranked_stocks = sorted(stocks, key=lambda x: x['score'], reverse=True)
            
            # 閫夋嫨鍓峃鍙?
            date_selected = ranked_stocks[:top_n]
            
            # 娣诲姞鎺掑悕
            for i, stock in enumerate(date_selected):
                stock['rank'] = i + 1
            
            selected_stocks_by_date[date] = date_selected
            total_selected += len(date_selected)
        
        # 鑾峰彇鏈€鏂版棩鏈熺殑閫夎偂浣滀负鍏煎鏃т唬鐮佺殑selected_stocks
        latest_date = max(stocks_by_date.keys()) if stocks_by_date else datetime.now().strftime('%Y%m%d')
        selected_stocks = selected_stocks_by_date.get(latest_date, [])
        
        # 缁勭粐缁撴灉 - 鍚屾椂淇濇寔鍏煎鎬у拰鏂扮粨鏋?
        result = {
            'selected_stocks': selected_stocks,  # 鍏煎鏃т唬鐮?
            'selected_stocks_by_date': selected_stocks_by_date,  # 鏂扮粨鏋勶細鎸夋棩鏈熷垎缁?
            'dates': sorted(stocks_by_date.keys()),
            'total_stocks': sum(len(stocks) for stocks in stocks_by_date.values()),
            'selection_criteria': '鍙樺寲鐐规娴?鐩镐技搴﹀尮閰?,
            'selection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': latest_date  # 鍏煎鏃т唬鐮?
        }
        
        self.logger.info(f"閫夎偂瀹屾垚锛屽叡澶勭悊浜?{len(stocks_by_date)} 涓氦鏄撴棩锛岄€夊嚭 {total_selected} 鍙偂绁?)
        return result

    def select_stocks_by_change_points(self, start_date=None, end_date=None, model_name="stock_selector", top_n=50, use_cache=True):
        """鍩轰簬鍙樺寲鐐规柟娉曢€夋嫨涓偂
        
        Args:
            start_date: 寮€濮嬫棩鏈燂紝濡傛灉涓篘one鍒欎娇鐢ㄩ粯璁ゅ€?
            end_date: 缁撴潫鏃ユ湡锛屽鏋滀负None鍒欎娇鐢ㄩ粯璁ゅ€?
            model_name: 妯″瀷鍚嶇О
            top_n: 杩斿洖鐨勮偂绁ㄦ暟閲?
            use_cache: 鏄惁浣跨敤鐗瑰緛缂撳瓨锛岄粯璁や负True
            
        Returns:
            dict: {'selected_stocks': [...], 'model_name': model_name, 'date': date}
        """
        # 璁剧疆鏃ユ湡鑼冨洿
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")
            
        # 纭繚浜ゆ槗鏃ュ巻宸插姞杞?
        if self.trading_days is None:
            self._load_trading_calendar()
            
        if not self.trading_days:
            self.logger.error("鏃犳硶鍔犺浇浜ゆ槗鏃ュ巻锛屾棤娉曠户缁?)
            return None
            
        # 鑾峰彇鏃ユ湡鑼冨洿鍐呯殑瀹為檯浜ゆ槗鏃?
        trading_days_in_range = [day for day in self.trading_days 
                               if start_date <= day <= end_date]
        
        if not trading_days_in_range:
            self.logger.error(f"鏃ユ湡鑼冨洿 {start_date} 鑷?{end_date} 鍐呮病鏈夋湁鏁堜氦鏄撴棩")
            return None
            
        # 浣跨敤瀹為檯浜ゆ槗鏃ユ潵鍒掑垎璁粌鍜岄娴嬫棩鏈?
        total_trading_days = len(trading_days_in_range)
        train_days_count = int(total_trading_days * 0.9)  # 浣跨敤60%鐨勪氦鏄撴棩浣滀负璁粌闆?
        
        train_end_idx = min(train_days_count, total_trading_days - 1)
        train_end_date = trading_days_in_range[train_end_idx]
        
        predict_start_idx = train_end_idx + 1
        predict_start_date = trading_days_in_range[predict_start_idx] if predict_start_idx < total_trading_days else end_date
        predict_end_date = end_date
        
        self.logger.info(f"鏃ユ湡鑼冨洿鍐呭叡鏈?{total_trading_days} 涓氦鏄撴棩")
        self.logger.info(f"璁粌鏃ユ湡鑼冨洿: {start_date} 鑷?{train_end_date} (鍏眥train_days_count}涓氦鏄撴棩)")
        self.logger.info(f"棰勬祴鏃ユ湡鑼冨洿: {predict_start_date} 鑷?{predict_end_date} (鍏眥total_trading_days - train_days_count}涓氦鏄撴棩)")
        
        # 姝ラ1: 鍔犺浇鏉垮潡鐗瑰緛
        self.logger.info("鍔犺浇鏉垮潡鐗瑰緛鏁版嵁...")
        sector_features_df = self._batch_load_features(self._get_all_sectors(), start_date, end_date)
        
        # 姝ラ2: 鍔犺浇涓偂鐗瑰緛
        self.logger.info("鍔犺浇涓偂鐗瑰緛鏁版嵁...")
        stock_features_df = self.load_stock_features(start_date=start_date, end_date=end_date)
        
        self.logger.info(f"鍒楋細{stock_features_df.columns}")
        
        if stock_features_df is None or stock_features_df.empty:
            self.logger.error("鏃犳硶鑾峰彇涓偂鐗瑰緛鏁版嵁")
            return None
        
        # 姝ラ3: 鍒掑垎璁粌闆嗗拰棰勬祴闆?
        # 绛涢€夎缁冩暟鎹?- 浣跨敤瀹為檯鐨勪氦鏄撴棩鏈熷垝鍒?
        train_mask = ((pd.to_datetime(stock_features_df['date_str']) >= pd.to_datetime(start_date)) & 
                      (pd.to_datetime(stock_features_df['date_str']) <= pd.to_datetime(train_end_date)))
        train_features = stock_features_df[train_mask].copy()
        
        # 绛涢€夐娴嬫暟鎹?
        predict_mask = ((pd.to_datetime(stock_features_df['date_str']) >= pd.to_datetime(predict_start_date)) & 
                         (pd.to_datetime(stock_features_df['date_str']) <= pd.to_datetime(predict_end_date)))
        predict_features = stock_features_df[predict_mask].copy()
        
        self.logger.info(f"璁粌鏁版嵁: {len(train_features)} 鏉¤褰?)
        self.logger.info(f"棰勬祴鏁版嵁: {len(predict_features)} 鏉¤褰?)
        
        # 淇濆瓨棰勬祴鏃ユ湡鍒楄〃锛岀敤浜庤繘搴﹁绠?
        predict_dates = sorted(predict_features['date_str'].unique())
        self.logger.info(f"棰勬祴鏁版嵁鍖呭惈 {len(predict_dates)} 涓氦鏄撴棩")
        
        # 纭繚棰勬祴鏁版嵁涓殑鏃ユ湡閮芥槸鏈夋晥浜ゆ槗鏃?
        valid_predict_dates = [date for date in predict_dates if date in self.trading_days]
        if len(valid_predict_dates) != len(predict_dates):
            self.logger.warning(f"棰勬祴鏁版嵁涓寘鍚?{len(predict_dates) - len(valid_predict_dates)} 涓潪浜ゆ槗鏃ワ紝灏嗚蹇界暐")
            # 鏇存柊棰勬祴鏁版嵁锛屽彧淇濈暀鏈夋晥浜ゆ槗鏃ョ殑鏁版嵁
            predict_features = predict_features[predict_features['date_str'].isin(valid_predict_dates)]
            predict_dates = valid_predict_dates
        
        
        # 姝ラ4: 妫€娴嬩釜鑲＄殑鍙樺寲鐐?
        self.logger.info("妫€娴嬩釜鑲″彉鍖栫偣...")
        stock_change_points = self._detect_stock_change_points(train_features)
        
        # 姝ラ5: 璁＄畻涓偂鐩稿鏉垮潡鐨勭壒寰侊紙浠呴拡瀵瑰彉鍖栫偣锛?
        self.logger.info("璁＄畻涓偂鐩稿鏉垮潡鐨勭壒寰侊紙浠呴拡瀵瑰彉鍖栫偣锛?..")
        train_features, train_relations = self.calculate_stock_sector_relative_features(
            train_features, sector_features_df, stock_change_points, is_training=True, use_cache=use_cache)
            
        # 瀵归娴嬫暟鎹篃杩涜鐩稿鐗瑰緛璁＄畻锛堝叏閮ㄦ棩鏈燂級
        self.logger.info("璁＄畻棰勬祴鏁版嵁鐨勪釜鑲＄浉瀵规澘鍧楃壒寰?..")
        predict_features, predict_relations = self.calculate_stock_sector_relative_features(
            predict_features, sector_features_df, is_training=False, use_cache=use_cache)
        
        # 姝ラ6: 鍒嗘瀽鍙樺寲鐐圭壒寰佹ā寮?
        self.logger.info("鍒嗘瀽涓偂鍙樺寲鐐圭壒寰佹ā寮?..")
        # 浼犻€掕缁冩暟鎹拰鏉垮潡鍏宠仈缁撴灉
        stock_feature_patterns = self._analyze_stock_change_point_patterns(train_features, stock_change_points, relations_df=train_relations)
        
        # 姝ラ7: 棰勬祴涓偂琛ㄧ幇
        self.logger.info("棰勬祴涓偂琛ㄧ幇...")
        # 浼犻€掗娴嬫暟鎹拰鏉垮潡鍏宠仈缁撴灉
        stock_predictions = self._predict_stocks_with_change_points(predict_features, stock_feature_patterns, relations_df=predict_relations)
        
        # 姝ラ8: 閫夋嫨琛ㄧ幇鏈€浣崇殑鑲＄エ
        self.logger.info(f"鎸変氦鏄撴棩鏈熼€夋嫨姣忓ぉ琛ㄧ幇鏈€浣崇殑鍓?{top_n} 鍙偂绁?..")
        selected_stocks = self._rank_and_select_stocks(stock_predictions, top_n)
        
        # 姝ラ9: 璇勪及閫夊嚭鑲＄エ鐨勫疄闄呰〃鐜?
        self.logger.info("璇勪及閫夊嚭鑲＄エ鐨勫疄闄呰〃鐜?..")
        evaluation_results = self._evaluate_selected_stocks(selected_stocks, stock_features_df)
        
        # 灏嗚瘎浼扮粨鏋滄坊鍔犲埌閫夎偂缁撴灉涓?
        selected_stocks['evaluation'] = evaluation_results
        selected_stocks['model_name'] = model_name
        selected_stocks['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 涓洪€夊嚭鐨勮偂绁ㄦ坊鍔犳帓鍚嶄俊鎭紙濡傛灉灏氭湭娣诲姞锛?
        if 'selected_stocks' in selected_stocks and selected_stocks['selected_stocks']:
            for i, stock in enumerate(selected_stocks['selected_stocks']):
                if 'rank' not in stock:
                    stock['rank'] = i + 1
        
        # 淇濆瓨缁撴灉
        result_path = self.model_dir / f"{model_name}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(selected_stocks, f, ensure_ascii=False, cls=NumpyEncoder, indent=2)
        
        # 淇濆瓨姣忔棩閫夎偂缁撴灉鍒板崟鐙枃浠?
        if 'selected_stocks_by_date' in selected_stocks and selected_stocks['selected_stocks_by_date']:
            daily_results_dir = self.model_dir / f"{model_name}_daily_results"
            os.makedirs(daily_results_dir, exist_ok=True)
            
            for date, stocks in selected_stocks['selected_stocks_by_date'].items():
                date_result_path = daily_results_dir / f"{date}_stocks.json"
                
                # 鑾峰彇璇ユ棩鏈熺殑璇勪及缁撴灉
                date_evaluation = None
                if 'evaluation' in selected_stocks and 'evaluation_by_date' in selected_stocks['evaluation']:
                    date_evaluation = selected_stocks['evaluation']['evaluation_by_date'].get(date)
                
                date_result = {
                    'selected_stocks': stocks,
                    'date': date,
                    'model_name': model_name,
                    'total_stocks': len(stocks),
                    'evaluation': date_evaluation
                }
                
                with open(date_result_path, "w", encoding="utf-8") as f:
                    json.dump(date_result, f, ensure_ascii=False, cls=NumpyEncoder, indent=2)
            
            self.logger.info(f"姣忔棩閫夎偂缁撴灉宸蹭繚瀛樺埌鐩綍: {daily_results_dir}")
        
        # 鐢熸垚璇勪及鎶ュ憡
        if 'evaluation' in selected_stocks:
            eval_results = selected_stocks['evaluation']
            
            # 澶勭悊鏂扮殑璇勪及缁撴灉鏍煎紡
            if 'overall_summary' in eval_results and isinstance(eval_results['overall_summary'], dict):
                summary = eval_results['overall_summary']
                
                # 杈撳嚭鏁翠綋璇勪及缁撴灉
                if 'avg_actual_return' in summary and 'win_rate' in summary:
                    self.logger.info(f"鏁翠綋閫夎偂璇勪及缁撴灉:")
                    self.logger.info(f"- 璇勪及鏃ユ湡鏁? {summary.get('total_evaluated_dates', 'N/A')}")
                    self.logger.info(f"- 璇勪及鑲＄エ鏁? {summary.get('total_evaluated_stocks', 'N/A')}")
                    self.logger.info(f"- 骞冲潎瀹為檯鏀剁泭鐜? {summary['avg_actual_return']:.2f}%")
                    self.logger.info(f"- 鏂瑰悜鍑嗙‘鐜? {summary['direction_accuracy']*100:.1f}%") 
                    self.logger.info(f"- 鑳滅巼: {summary['win_rate']*100:.1f}%")
                    self.logger.info(f"- 琛ㄧ幇璇勭骇: {summary['performance_rating']}")
                    
                    # 濡傛灉鏈夋棩骞冲潎琛ㄧ幇鏁版嵁
                    if 'avg_daily_return' in summary:
                        self.logger.info(f"- 鏃ュ潎鏀剁泭鐜? {summary['avg_daily_return']:.2f}%")
                        self.logger.info(f"- 鏃ュ潎鑳滅巼: {summary['avg_daily_win_rate']*100:.1f}%")
            
            # 澶勭悊鏃х殑璇勪及缁撴灉鏍煎紡 (鍚戝悗鍏煎)
            elif 'evaluation_summary' in eval_results and isinstance(eval_results['evaluation_summary'], dict):
                summary = eval_results['evaluation_summary']
                if 'avg_actual_return' in summary and 'win_rate' in summary and 'direction_accuracy' in summary:
                    self.logger.info(f"閫夎偂璇勪及缁撴灉:")
                    self.logger.info(f"- 骞冲潎瀹為檯鏀剁泭鐜? {summary['avg_actual_return']:.2f}%")
                    self.logger.info(f"- 鏂瑰悜鍑嗙‘鐜? {summary['direction_accuracy']*100:.1f}%") 
                    self.logger.info(f"- 鑳滅巼: {summary['win_rate']*100:.1f}%")
                    self.logger.info(f"- 琛ㄧ幇璇勭骇: {summary['performance_rating']}")
        
        self.logger.info(f"閫夎偂缁撴灉鍜岃瘎浼板凡淇濆瓨鍒?{result_path}")
        
        return selected_stocks

    def load_stock_features(self, start_date=None, end_date=None):
        """
        鍔犺浇涓偂鐗瑰緛鏁版嵁锛屼娇鐢ㄧ紦瀛樻満鍒舵彁楂樻晥鐜?
        
        Args:
            start_date: 寮€濮嬫棩鏈燂紝鏍煎紡涓篩YYYMMDD
            end_date: 缁撴潫鏃ユ湡锛屾牸寮忎负YYYYMMDD
            
        Returns:
            DataFrame: 鑲＄エ鐗瑰緛鏁版嵁
        """
        self.logger.info(f"鍔犺浇涓偂鐗瑰緛鏁版嵁 {start_date} 鑷?{end_date}...")
        
        # 鍒涘缓缂撳瓨閿?
        cache_key = f"stock_features_{start_date}_{end_date}"
        cache_file = self.features_cache_dir / f"{cache_key}.parquet"
        
        # 妫€鏌ョ紦瀛?
        if cache_file.exists():
            self.logger.info(f"浠庣紦瀛樺姞杞戒釜鑲＄壒寰佹暟鎹? {cache_file}")
            cached_data = pd.read_parquet(cache_file)
            
            # 妫€鏌ョ紦瀛樻槸鍚﹀寘鍚繀瑕佺殑鍒?
            required_columns = ['future_return_2d']
            missing_columns = [col for col in required_columns if col not in cached_data.columns]
            
            if not missing_columns:
                # 妫€鏌uture_return_2d鍒楁槸鍚︽湁瓒冲鐨勬湁鏁堝€?
                valid_pct = cached_data['future_return_2d'].notna().sum() / len(cached_data)
                self.logger.info(f"缂撳瓨鏁版嵁涓璮uture_return_2d鍒楁湁鏁堝€兼瘮渚? {valid_pct:.2f}")
                
                if valid_pct >= 0.5:  # 濡傛灉鏈夋晥鍊兼瘮渚嬭冻澶熼珮
                    self.logger.info(f"缂撳瓨鏁版嵁鍖呭惈鎵€鏈夊繀瑕佸垪涓旀湁鏁堬紝鐩存帴浣跨敤")
                    return cached_data
                else:
                    self.logger.warning(f"缂撳瓨鏁版嵁涓璮uture_return_2d鍒楁湁鏁堝€兼瘮渚嬭繃浣? {valid_pct:.2f}锛岄渶瑕侀噸鏂拌绠?)
            else:
                self.logger.warning(f"缂撳瓨鏁版嵁缂哄皯蹇呰鍒? {missing_columns}锛岄渶瑕侀噸鏂拌绠?)
        
        # 缂撳瓨涓嶅瓨鍦ㄦ垨鑰呬笉瀹屾暣锛屼粠鍘熷鐗瑰緛鏂囦欢鍔犺浇
        try:
            # 纭畾鏃ユ湡鑼冨洿
            if start_date is None or end_date is None:
                raise ValueError("蹇呴』鎸囧畾寮€濮嬫棩鏈熷拰缁撴潫鏃ユ湡")
                
            # 鑾峰彇鏃ユ湡鑼冨洿鍐呯殑鎵€鏈変氦鏄撴棩
            calendar = self._load_trading_calendar()
            if calendar is None:
                raise ValueError("鏃犳硶鍔犺浇浜ゆ槗鏃ュ巻")
                
            # 绛涢€夋棩鏈熻寖鍥村唴鐨勪氦鏄撴棩
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            trading_dates = [d for d in calendar if start_dt <= pd.to_datetime(d) <= end_dt]
            
            if not trading_dates:
                raise ValueError(f"鎸囧畾鐨勬棩鏈熻寖鍥?{start_date} 鑷?{end_date} 鍐呮病鏈変氦鏄撴棩")
            
            # 浠巊enerate_daily_features.py涓幏鍙栫殑淇濆瓨鐗瑰緛鐨勭洰褰曡矾寰?
            features_dir = get_data_path("features/daily")
            
            # 浠庢瘡涓氦鏄撴棩鐨勭壒寰佹枃浠跺姞杞芥暟鎹?
            all_data = []
            for date in trading_dates:
                date_str = date if isinstance(date, str) else date.strftime("%Y%m%d")
                feature_file = features_dir / f"{date_str}.parquet"
                
                if not os.path.exists(feature_file):
                    self.logger.warning(f"鏃ユ湡 {date_str} 鐨勭壒寰佹枃浠朵笉瀛樺湪: {feature_file}")
                    continue
                    
                try:
                    # 鍔犺浇褰撴棩鐗瑰緛
                    daily_features = pd.read_parquet(feature_file)
                    all_data.append(daily_features)
                    self.logger.debug(f"鎴愬姛鍔犺浇鏃ユ湡 {date_str} 鐨勭壒寰佹暟鎹紝鍏?{len(daily_features)} 鏉¤褰?)
                except Exception as e:
                    self.logger.error(f"鍔犺浇鏃ユ湡 {date_str} 鐨勭壒寰佹暟鎹椂鍑洪敊: {e}")
                    continue
            
            if not all_data:
                raise ValueError(f"鏈兘鍔犺浇浠讳綍鏃ユ湡鐨勭壒寰佹暟鎹?)
                
            # 鍚堝苟鎵€鏈夋棩鏈熺殑鏁版嵁
            combined_features = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"鎴愬姛鍔犺浇 {len(trading_dates)} 涓氦鏄撴棩鐨勭壒寰佹暟鎹紝鍏?{len(combined_features)} 鏉¤褰?)
            
            # 瀵圭壒寰佽繘琛岄珮绾у垎鏋愬拰澧炲己澶勭悊
            enhanced_features = self._enhance_stock_features(combined_features)
            
            # 淇濆瓨鍒扮紦瀛?
            ensure_dir(self.features_cache_dir)
            enhanced_features.to_parquet(cache_file)
            self.logger.info(f"澧炲己鍚庣殑涓偂鐗瑰緛鏁版嵁宸茬紦瀛樺埌: {cache_file}")
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"鍔犺浇涓偂鐗瑰緛鏁版嵁鏃跺嚭閿? {e}")
            self.logger.error(traceback.format_exc())
            raise ValueError(f"鍔犺浇涓偂鐗瑰緛鏁版嵁澶辫触: {e}")
    
    def _enhance_stock_features(self, features_df):
        """
        澧炲己鑲＄エ鐗瑰緛锛屾坊鍔犳妧鏈垎鏋愭寚鏍囩瓑
        """
        if features_df is None or features_df.empty:
            self.logger.warning("鐗瑰緛鏁版嵁涓虹┖锛屾棤娉曞寮虹壒寰?)
            return features_df
            
        self.logger.info(f"寮€濮嬪寮鸿偂绁ㄧ壒寰侊紝娣诲姞鎶€鏈垎鏋愭寚鏍?..")
        # self.logger.debug(f"鍘熷鐗瑰緛鍒? {features_df.columns.tolist()}")
            
        # 纭繚鎸夎偂绁ㄥ拰鏃ユ湡鎺掑簭
        features_df = features_df.sort_values(['stock_code', 'date_str'])
        
        # 娑ㄨ穼鍋滅浉鍏宠缃?
        # 娌繁涓ゅ競鑲＄エ鐨勬定璺屽仠姣斾緥鍦ㄤ笉鍚屾椂鏈熸湁璋冩暣
        # 杩欓噷浣跨敤閫氱敤璁剧疆 - 瀹為檯搴旂敤涓簲褰撴牴鎹偂绁ㄧ被鍨嬪拰浜ゆ槗鏃ユ湡鍔ㄦ€佽缃?
        up_limit_ratio = 0.1   # 鏅€氫釜鑲℃定鍋滄瘮渚?0%
        down_limit_ratio = -0.1  # 鏅€氫釜鑲¤穼鍋滄瘮渚?10%
        
        # 鍒涚鏉?ST鑲＄瓑鍙兘鏈変笉鍚岀殑娑ㄨ穼鍋滄瘮渚?
        # 绠€鍖栧鐞? 鍒涗笟鏉?绉戝垱鏉跨瓑涓?0%锛孲T鑲′负5%
        for idx, row in features_df.iterrows():
            # 鑲＄エ浠ｇ爜棣栧瓧绗︿负6涓旂浜屽瓧绗︿负8鐨勪负绉戝垱鏉?
            if row['stock_code'].startswith('68'):
                up_limit_ratio = 0.2
                down_limit_ratio = -0.2
            # 鑲＄エ浠ｇ爜棣栧瓧绗︿负3鐨勪负鍒涗笟鏉?
            elif row['stock_code'].startswith('3'):
                up_limit_ratio = 0.2  
                down_limit_ratio = -0.2
            elif '.BJ' in row['stock_code']:  # 鍖椾氦鎵€鑲＄エ
                up_limit_ratio = 0.3
                down_limit_ratio = -0.3
            # 鍖呭惈ST鐨勬槸鐗瑰埆澶勭悊鑲＄エ
            elif 'ST' in str(row.get('stock_name', '')):
                up_limit_ratio = 0.05
                down_limit_ratio = -0.05
            else:
                up_limit_ratio = 0.1
                down_limit_ratio = -0.1
                
            # 鍒ゆ柇鏄惁娑ㄥ仠鎴栬穼鍋?
            if 'close' in row.index and 'pre_close' in row.index:
                # 璁＄畻鐞嗚娑ㄥ仠浠?
                theoretical_limit_up = row['pre_close'] * (1 + up_limit_ratio)
                # 鍥涜垗浜斿叆鍒?.01鍏?
                rounded_limit_up = round(theoretical_limit_up, 2)
                # 鍒ゆ柇鏄惁娑ㄥ仠
                features_df.loc[idx, 'is_limit_up'] = abs(row['close'] - rounded_limit_up) < 0.005
                
                # 璁＄畻鐞嗚璺屽仠浠?
                theoretical_limit_down = row['pre_close'] * (1 + down_limit_ratio)
                # 鍥涜垗浜斿叆鍒?.01鍏?
                rounded_limit_down = round(theoretical_limit_down, 2)
                # 鍒ゆ柇鏄惁璺屽仠
                features_df.loc[idx, 'is_limit_down'] = abs(row['close'] - rounded_limit_down) < 0.005
            elif 'change_percent' in row.index:
                # 濡傛灉鏈彁渚涗环鏍兼暟鎹紝鍒欎娇鐢ㄦ定璺屽箙鍒ゆ柇锛屽厑璁镐竴瀹氳宸?
                up_limit_pct = up_limit_ratio * 100
                down_limit_pct = down_limit_ratio * 100
                features_df.loc[idx, 'is_limit_up'] = abs(row['change_percent'] - up_limit_pct) <= 0.15
                features_df.loc[idx, 'is_limit_down'] = abs(row['change_percent'] - down_limit_pct) <= 0.15
        
        # 鎸夎偂绁ㄥ垎缁勫鐞?
        grouped = features_df.groupby('stock_code')
        stock_groups = list(grouped)
        
        # 浣跨敤骞惰澶勭悊锛屼絾閬垮厤pickling閿欒
        import pandas as pd
        from tqdm import tqdm
        import multiprocessing
        
        # 纭畾鏄惁浣跨敤骞惰澶勭悊
        use_parallel = len(stock_groups) > 3 and multiprocessing.cpu_count() > 2
        all_enhanced = []
        
        if use_parallel:
            # 浼樺厛浣跨敤杩涚▼姹犱互鎻愰珮鎬ц兘锛屽鏋滃嚭閿欏垯閫€鍥炲埌绾跨▼姹?
            try:
                from concurrent.futures import ProcessPoolExecutor
                num_cpus = multiprocessing.cpu_count()
                max_workers = min(num_cpus - 1, 8)  # 淇濈暀1涓狢PU鏍稿績缁欑郴缁燂紝鏈€澶氫娇鐢?涓繘绋?
                
                self.logger.info(f"浣跨敤骞惰杩涚▼澶勭悊澧炲己 {len(stock_groups)} 鍙偂绁ㄧ殑鐗瑰緛锛屼娇鐢?{max_workers} 涓繘绋?..")
                
                # 鍒涘缓澶勭悊鍑芥暟
                def process_stock_wrapper(group_tuple):
                    stock_code, stock_df = group_tuple
                    try:
                        # 纭繚stock_df鏄竴涓狣ataFrame
                        if not isinstance(stock_df, pd.DataFrame):
                            self.logger.error(f"鑲＄エ {stock_code} 鐨勬暟鎹笉鏄疍ataFrame")
                            return None
                        return self._process_single_stock_features(stock_df)
                    except Exception as e:
                        self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鏃跺嚭閿? {str(e)}")
                        return stock_df
                
                # 浣跨敤杩涚▼姹犳墽琛?
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = []
                    for stock_code, stock_df in tqdm(stock_groups, desc="澶勭悊鑲＄エ鐗瑰緛"):
                        results.append(executor.submit(process_stock_wrapper, (stock_code, stock_df)))
                    
                    # 鏀堕泦缁撴灉 - 纭繚姝ｇ‘鑾峰彇Future鐨勭粨鏋?
                    for future in tqdm(results, desc="鏀堕泦澶勭悊缁撴灉"):
                        result = future.result()
                        if result is not None:
                            all_enhanced.append(result)
            
            except Exception as e:
                self.logger.warning(f"浣跨敤杩涚▼姹犲鐞嗗け璐ワ紝鍒囨崲鍒扮嚎绋嬫睜: {e}")
                
                # 濡傛灉杩涚▼姹犲け璐ワ紝閫€鍥炲埌绾跨▼姹?
                from concurrent.futures import ThreadPoolExecutor
                
                self.logger.info(f"浣跨敤绾跨▼姹犲鐞?{len(stock_groups)} 鍙偂绁ㄧ殑鐗瑰緛...")
                
                # 娓呯┖涔嬪墠鐨勭粨鏋?
                all_enhanced = []
                
                # 浣跨敤绾跨▼姹?
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for stock_code, stock_df in tqdm(stock_groups, desc="澶勭悊鑲＄エ鐗瑰緛"):
                        try:
                            enhanced_df = self._process_single_stock_features(stock_df)
                            all_enhanced.append(enhanced_df)
                        except Exception as e:
                            self.logger.error(f"绾跨▼姹犲鐞嗚偂绁?{stock_code} 鏃跺嚭閿? {str(e)}")
                            all_enhanced.append(stock_df)  # 濡傛灉澶勭悊澶辫触锛屼娇鐢ㄥ師濮嬫暟鎹?
        else:
            self.logger.info(f"鑲＄エ鏁伴噺杈冨皯鎴朇PU鏍稿績鏁颁笉瓒筹紝浣跨敤椤哄簭澶勭悊 {len(stock_groups)} 鍙偂绁?..")
            for stock_code, stock_df in tqdm(stock_groups, desc="澶勭悊鑲＄エ鐗瑰緛"):
                try:
                    enhanced_df = self._process_single_stock_features(stock_df)
                    all_enhanced.append(enhanced_df)
                except Exception as e:
                    self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鏃跺嚭閿? {str(e)}")
                    all_enhanced.append(stock_df)  # 濡傛灉澶勭悊澶辫触锛屼娇鐢ㄥ師濮嬫暟鎹?
        
        # 鍚堝苟鎵€鏈夊寮哄悗鐨勬暟鎹?
        if all_enhanced:
            result_df = pd.concat(all_enhanced, ignore_index=False)
            self.logger.info(f"鐗瑰緛澧炲己瀹屾垚锛屾坊鍔犱簡褰㈡€佽瘑鍒拰鎶€鏈垎鏋愮壒寰?)
            
            # 妫€鏌uture_return_2d鍒楁槸鍚﹀瓨鍦ㄥ苟鍖呭惈鏈夋晥鍊?
            if 'future_return_2d' in result_df.columns:
                valid_pct = result_df['future_return_2d'].notna().sum() / len(result_df)
                self.logger.info(f"future_return_2d鍒楁湁鏁堝€兼瘮渚? {valid_pct:.2f}")
                if valid_pct < 0.5:
                    self.logger.warning(f"future_return_2d鍒楁湁鏁堝€兼瘮渚嬭繃浣? {valid_pct:.2f}锛屽彲鑳藉瓨鍦ㄨ绠楅棶棰?)
            else:
                self.logger.warning(f"缁撴灉涓己灏慺uture_return_2d鍒楋紝杩欏彲鑳戒細瀵艰嚧鍚庣画鍒嗘瀽鍑洪敊")
            
            self.logger.debug(f"澧炲己鍚庣殑鐗瑰緛鍒? {result_df.columns.tolist()}")
            return result_df
        else:
            self.logger.warning("娌℃湁鎴愬姛澧炲己浠讳綍鑲＄エ鐗瑰緛")
            return features_df

    def _process_single_stock_features(self, stock_df):
        """澶勭悊鍗曞彧鑲＄エ鐨勭壒寰侊紝鐢ㄤ簬骞惰澶勭悊"""
        try:
            self.logger.info(f"寮€濮嬪鐞嗚偂绁?{stock_df['stock_code'].iloc[0]} 鐨勭壒寰?..")
            # 娣诲姞K绾挎ā寮忚瘑鍒?
            enhanced_df = self._detect_candlestick_patterns(stock_df)
            
            # 娣诲姞鍥捐〃褰㈡€佽瘑鍒?
            enhanced_df = self._detect_chart_patterns(enhanced_df)
            
            # 鍒嗘瀽浠锋牸瓒嬪娍
            enhanced_df = self._analyze_price_trend(enhanced_df)
            
            # 璁＄畻椋庨櫓鏀剁泭姣旂巼
            enhanced_df = self._calculate_risk_reward_ratios(enhanced_df)
            
            # 娣诲姞缁煎悎鐘舵€佹爣绛?
            enhanced_df = self._generate_state_labels(enhanced_df)
            
            # 璁＄畻future_return_2d鐗瑰緛 - 娣诲姞鏈潵2鏃ユ敹鐩婄巼
            enhanced_df = self._calculate_future_returns(enhanced_df)
            
            return enhanced_df
        except Exception as e:
            self.logger.error(f"澶勭悊鑲＄エ {stock_df['stock_code'].iloc[0]} 鐗瑰緛鏃跺嚭閿? {str(e)}")
            return stock_df  # 杩斿洖鍘熷鏁版嵁妗?

    def _calculate_future_returns(self, df):
        """
        璁＄畻涓偂鏈潵鏀剁泭鐜囩壒寰?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞浜嗘湭鏉ユ敹鐩婄巼鐨勬暟鎹?
        """
        stock_code = df['stock_code'].iloc[0] if 'stock_code' in df.columns and len(df) > 0 else "鏈煡"
        self.logger.info(f"璁＄畻鑲＄エ {stock_code} 鐨勬湭鏉ユ敹鐩婄巼...")
        
        # 鍒濆鍖杅uture_return_2d鍒?
        df['future_return_2d'] = np.nan
        
        try:
            # 纭繚浜ゆ槗鏃ュ巻宸插姞杞?
            if self.trading_days is None or not self.trading_days:
                self._load_trading_calendar()
            
            if self.trading_days is None or not self.trading_days:
                self.logger.error("浜ゆ槗鏃ュ巻涓虹┖鎴栧姞杞藉け璐ワ紝鏃犳硶璁＄畻鏈潵鏀剁泭鐜?)
                return df
                
            # 纭繚鎸夋棩鏈熸帓搴?
            if 'date_str' not in df.columns:
                self.logger.warning(f"鑲＄エ {stock_code} 娌℃湁date_str鍒楋紝鏃犳硶璁＄畻鏈潵鏀剁泭鐜?)
                return df
                
            df = df.sort_values('date_str')
            
            # 妫€鏌ユ槸鍚︽湁close鍒?
            if 'close' not in df.columns:
                self.logger.warning(f"鑲＄エ {stock_code} 娌℃湁close鍒楋紝鏃犳硶璁＄畻鏈潵鏀剁泭鐜?)
                return df
                
            # 鑾峰彇鎵€鏈夋棩鏈熷拰瀹冧滑鍦ㄤ氦鏄撴棩鍘嗕腑鐨勭储寮?
            date_indices = {}
            valid_dates = 0
            
            for i, date_str in enumerate(df['date_str']):
                # 纭繚鏃ユ湡瀛楃涓叉牸寮忔纭?
                try:
                    if isinstance(date_str, pd.Timestamp):
                        date_str = date_str.strftime('%Y%m%d')
                    elif not isinstance(date_str, str):
                        date_str = str(date_str)
                        
                    # 濡傛灉鏃ユ湡鍦ㄤ氦鏄撴棩鍘嗕腑锛岃褰曞叾绱㈠紩
                    if date_str in self.trading_days:
                        date_indices[i] = self.trading_days.index(date_str)
                        valid_dates += 1
                except Exception as e:
                    self.logger.warning(f"澶勭悊鏃ユ湡 {date_str} 鏃跺嚭閿? {e}")
                    continue
            
            if valid_dates == 0:
                self.logger.warning(f"鑲＄エ {stock_code} 娌℃湁鏈夋晥鐨勪氦鏄撴棩鏈燂紝鏃犳硶璁＄畻鏈潵鏀剁泭鐜?)
                return df
                
            self.logger.info(f"鎵惧埌 {valid_dates} 涓湁鏁堜氦鏄撴棩鏈?)
            
            # 璁＄畻鏈潵2澶╂敹鐩婄巼
            calculated_returns = 0
            for i in range(len(df)):
                if i not in date_indices:
                    continue
                    
                current_idx = date_indices[i]
                
                # 纭繚褰撳墠琛屾湁鏁?
                try:
                    current_close = df.iloc[i]['close']
                    
                    if pd.isna(current_close) or current_close <= 0:
                        continue
                        
                    # 妫€鏌ユ槸鍚︽湁瓒冲鐨勬湭鏉ヤ氦鏄撴棩
                    if current_idx + 2 < len(self.trading_days):
                        future_date_str = self.trading_days[current_idx + 2]
                        future_row = df[df['date_str'] == future_date_str]
                        
                        if not future_row.empty:
                            future_close = future_row.iloc[0]['close']
                            if not pd.isna(future_close) and future_close > 0:
                                # 璁＄畻鏀剁泭鐜?
                                future_return = (future_close - current_close) / current_close
                                df.iloc[i, df.columns.get_loc('future_return_2d')] = future_return
                                calculated_returns += 1
                except Exception as e:
                    self.logger.warning(f"璁＄畻绗?{i} 琛岀殑鏈潵鏀剁泭鐜囨椂鍑洪敊: {e}")
                    continue
                            
            valid_ratio = df['future_return_2d'].notna().sum() / len(df) if len(df) > 0 else 0
            self.logger.info(f"鑲＄エ {stock_code} 鐨勬湭鏉ユ敹鐩婄巼璁＄畻瀹屾垚锛岃绠椾簡 {calculated_returns} 涓€硷紝鏈夋晥鍊兼瘮渚? {valid_ratio:.2f}")
            
            # 妫€鏌ユ槸鍚︽病鏈夎绠楀嚭浠讳綍鏈夋晥鍊?
            if calculated_returns == 0:
                self.logger.warning(f"鑲＄エ {stock_code} 鏈兘璁＄畻鍑轰换浣曟湁鏁堢殑鏈潵鏀剁泭鐜囧€?)
                
        except Exception as e:
            self.logger.error(f"璁＄畻鑲＄エ {stock_code} 鐨勬湭鏉ユ敹鐩婄巼鏃跺嚭閿? {e}")
            self.logger.error(traceback.format_exc())
            
        return df
        
    def _detect_candlestick_patterns(self, df):
        """
        妫€娴婯绾垮舰鎬?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞K绾垮舰鎬佺壒寰佺殑鏁版嵁
        """
        # 鍒濆鍖栧舰鎬佸垪
        pattern_columns = [
            'pattern_doji', 'pattern_hammer', 'pattern_inverted_hammer', 
            'pattern_engulfing_bullish', 'pattern_engulfing_bearish',
            'pattern_morning_star', 'pattern_evening_star',
            'pattern_three_white_soldiers', 'pattern_three_black_crows',
            'pattern_hanging_man', 'pattern_shooting_star'
        ]
        
        for col in pattern_columns:
            df[col] = 0
            
        # 鎻愬彇浠锋牸鏁版嵁
        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        
        # 璁＄畻瀹炰綋澶у皬鍜屽奖绾?
        body_size = np.abs(close_price - open_price)
        upper_shadow = high_price - np.maximum(open_price, close_price)
        lower_shadow = np.minimum(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        # 閬垮厤闄ら浂閿欒
        total_range = np.where(total_range == 0, 0.001, total_range)
        
        # 1. 鍗佸瓧鏄?Doji) - 寮€鐩樹环鍜屾敹鐩樹环鍑犱箮鐩稿悓
        df['pattern_doji'] = np.where(body_size / total_range < 0.1, 1, 0)
        
        # 2. 閿ゅ瓙绾?Hammer) - 涓嬪奖绾块暱锛屼笂褰辩嚎鐭紝瀹炰綋灏?
        df['pattern_hammer'] = np.where(
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < 0.2 * total_range) & 
            (body_size / total_range < 0.3),
            1, 0
        )
        
        # 3. 鍊掗敜瀛愮嚎(Inverted Hammer)
        df['pattern_inverted_hammer'] = np.where(
            (upper_shadow > 2 * body_size) & 
            (lower_shadow < 0.2 * total_range) & 
            (body_size / total_range < 0.3),
            1, 0
        )
        
        # 4. 鍚炲櫖褰㈡€?Engulfing Pattern)
        for i in range(1, len(df)):
            # 4.1 鐪嬫定鍚炲櫖
            if (close_price[i] > open_price[i] and  # 褰撴棩闃崇嚎
                close_price[i-1] < open_price[i-1] and  # 鍓嶆棩闃寸嚎
                close_price[i] > open_price[i-1] and  # 褰撴棩鏀剁洏浠烽珮浜庡墠鏃ュ紑鐩樹环
                open_price[i] < close_price[i-1]):  # 褰撴棩寮€鐩樹环浣庝簬鍓嶆棩鏀剁洏浠?
                df.iloc[i, df.columns.get_loc('pattern_engulfing_bullish')] = 1
                
            # 4.2 鐪嬭穼鍚炲櫖
            if (close_price[i] < open_price[i] and  # 褰撴棩闃寸嚎
                close_price[i-1] > open_price[i-1] and  # 鍓嶆棩闃崇嚎
                close_price[i] < open_price[i-1] and  # 褰撴棩鏀剁洏浠蜂綆浜庡墠鏃ュ紑鐩樹环
                open_price[i] > close_price[i-1]):  # 褰撴棩寮€鐩樹环楂樹簬鍓嶆棩鏀剁洏浠?
                df.iloc[i, df.columns.get_loc('pattern_engulfing_bearish')] = 1
        
        # 缁煎悎褰㈡€佸己搴?
        df['candlestick_pattern_strength'] = df[pattern_columns].sum(axis=1)
        
        return df
        
    def _detect_chart_patterns(self, df):
        """
        妫€娴嬬粡鍏稿浘琛ㄥ舰鎬侊紝濡傚ご鑲╅《/搴曘€佸弻椤?鍙屽簳銆佷笁瑙掑舰绛?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞鍥捐〃褰㈡€佺壒寰佺殑鏁版嵁
        """
        # 鍒濆鍖栧舰鎬佸垪
        pattern_columns = [
            'pattern_head_shoulders', 'pattern_inverse_head_shoulders',
            'pattern_double_top', 'pattern_double_bottom',
            'pattern_triangle_ascending', 'pattern_triangle_descending',
            'pattern_triangle_symmetrical', 'pattern_flag_bullish', 'pattern_flag_bearish'
        ]
        
        for col in pattern_columns:
            df[col] = 0
        
        # 妫€娴嬬獥鍙ｅぇ灏?- 澧炲ぇ绐楀彛浠ユ洿濂藉湴鎹曟崏澶嶆潅褰㈡€?
        window_size = 30
        if len(df) < window_size:
            return df
        
        # 鎻愬彇浠锋牸鏁版嵁
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        # 浣跨敤婊戝姩绐楀彛妫€娴嬪舰鎬?
        for i in range(window_size, len(df)):
            try:
                window_close = close[i-window_size:i+1]
                window_high = high[i-window_size:i+1]
                window_low = low[i-window_size:i+1]
                window_volume = volume[i-window_size:i+1] if volume is not None else None
                
                # 鏌ユ壘灞€閮ㄦ瀬鍊肩偣(宄板拰璋?
                peaks = self._find_peaks(window_close)
                troughs = self._find_peaks(-window_close)  # 閫氳繃鍙栧弽鎵惧埌浣庤胺
                
                # 1. 妫€娴嬪ご鑲╅《褰㈡€?(鐪嬭穼鍙嶈浆褰㈡€?
                # 澶磋偐椤堕渶瑕佷笁涓嘲锛屼腑闂寸殑宄?澶撮儴)鏈€楂橈紝涓や晶鐨勫嘲(鑲╅儴)楂樺害鐩歌繎
                if len(peaks) >= 3:
                    # 鑾峰彇鏈€杩戠殑涓変釜宄?
                    last_three_peaks = sorted(peaks[-3:])
                    if len(last_three_peaks) == 3:
                        left_shoulder, head, right_shoulder = last_three_peaks
                        
                        # 纭繚澶撮儴楂樹簬鑲╅儴
                        if (window_close[head] > window_close[left_shoulder] and 
                            window_close[head] > window_close[right_shoulder]):
                            
                            # 纭繚涓よ偐楂樺害鐩歌繎 (鍏佽10%鐨勫樊寮?
                            shoulder_diff = abs(window_close[left_shoulder] - window_close[right_shoulder])
                            if shoulder_diff / window_close[left_shoulder] < 0.1:
                                
                                # 纭繚鏈夐绾?(涓よ偐涔嬮棿鐨勪綆鐐硅繛绾?
                                # 鑷冲皯闇€瑕佷竴涓綆鐐瑰湪宸﹁偐鍜屽ご閮ㄤ箣闂达紝涓€涓湪澶撮儴鍜屽彸鑲╀箣闂?
                                left_troughs = [t for t in troughs if left_shoulder < t < head]
                                right_troughs = [t for t in troughs if head < t < right_shoulder]
                                
                                if left_troughs and right_troughs:
                                    # 棰堢嚎搴旇澶ц嚧姘村钩
                                    neck_left = window_close[left_troughs[-1]]
                                    neck_right = window_close[right_troughs[0]]
                                    
                                    if abs(neck_left - neck_right) / neck_left < 0.05:
                                        # 妫€鏌ユ渶鍚庝竴涓环鏍兼槸鍚︽帴杩戞垨浣庝簬棰堢嚎锛岃〃绀哄舰鎬佸凡瀹屾垚
                                        if window_close[-1] <= neck_right * 1.01:
                                            df.iloc[i, df.columns.get_loc('pattern_head_shoulders')] = 1
                
                # 2. 妫€娴嬪ご鑲╁簳褰㈡€?(鐪嬫定鍙嶈浆褰㈡€?
                if len(troughs) >= 3:
                    # 鑾峰彇鏈€杩戠殑涓変釜浣庤胺
                    last_three_troughs = sorted(troughs[-3:])
                    if len(last_three_troughs) == 3:
                        left_shoulder, head, right_shoulder = last_three_troughs
                        
                        # 纭繚澶撮儴浣庝簬鑲╅儴
                        if (window_close[head] < window_close[left_shoulder] and 
                            window_close[head] < window_close[right_shoulder]):
                            
                            # 纭繚涓よ偐楂樺害鐩歌繎
                            shoulder_diff = abs(window_close[left_shoulder] - window_close[right_shoulder])
                            if shoulder_diff / window_close[left_shoulder] < 0.1:
                                
                                # 纭繚鏈夐绾?(涓よ偐涔嬮棿鐨勯珮鐐硅繛绾?
                                left_peaks = [p for p in peaks if left_shoulder < p < head]
                                right_peaks = [p for p in peaks if head < p < right_shoulder]
                                
                                if left_peaks and right_peaks:
                                    # 棰堢嚎搴旇澶ц嚧姘村钩
                                    neck_left = window_close[left_peaks[-1]]
                                    neck_right = window_close[right_peaks[0]]
                                    
                                    if abs(neck_left - neck_right) / neck_left < 0.05:
                                        # 妫€鏌ユ渶鍚庝竴涓环鏍兼槸鍚︽帴杩戞垨楂樹簬棰堢嚎锛岃〃绀哄舰鎬佸凡瀹屾垚
                                        if window_close[-1] >= neck_right * 0.99:
                                            df.iloc[i, df.columns.get_loc('pattern_inverse_head_shoulders')] = 1
                
                # 3. 妫€娴嬪弻椤跺舰鎬?(鐪嬭穼鍙嶈浆褰㈡€?
                if len(peaks) >= 2:
                    # 鑾峰彇鏈€杩戠殑涓や釜宄?
                    last_two_peaks = sorted(peaks[-2:])
                    if len(last_two_peaks) == 2:
                        peak1, peak2 = last_two_peaks
                        
                        # 纭繚涓や釜椤剁偣楂樺害鐩歌繎 (鍏佽3%鐨勫樊寮?
                        if abs(window_close[peak1] - window_close[peak2]) / window_close[peak1] < 0.03:
                            # 纭繚涓ゅ嘲涔嬮棿鏈夋槑鏄剧殑浣庤胺
                            between_troughs = [t for t in troughs if peak1 < t < peak2]
                            if between_troughs:
                                trough = between_troughs[0]  # 浣跨敤鏈€鏄捐憲鐨勪綆璋?
                                
                                # 纭繚浣庤胺鏄捐憲浣庝簬涓や釜椤剁偣
                                if (window_close[trough] < window_close[peak1] * 0.97 and
                                    window_close[trough] < window_close[peak2] * 0.97):
                                    
                                    # 妫€鏌ュ舰鎬佹槸鍚﹀凡瀹屾垚 - 浠锋牸璺岀牬棰堢嚎(鏀拺浣?
                                    if window_close[-1] < window_close[trough]:
                                        df.iloc[i, df.columns.get_loc('pattern_double_top')] = 1
            
                # 4. 妫€娴嬪弻搴曞舰鎬?(鐪嬫定鍙嶈浆褰㈡€?
                if len(troughs) >= 2:
                    # 鑾峰彇鏈€杩戠殑涓や釜璋?
                    last_two_troughs = sorted(troughs[-2:])
                    if len(last_two_troughs) == 2:
                        trough1, trough2 = last_two_troughs
                        
                        # 纭繚涓や釜搴曠偣楂樺害鐩歌繎
                        if abs(window_close[trough1] - window_close[trough2]) / window_close[trough1] < 0.03:
                            # 纭繚涓よ胺涔嬮棿鏈夋槑鏄剧殑楂樼偣
                            between_peaks = [p for p in peaks if trough1 < p < trough2]
                            if between_peaks:
                                peak = between_peaks[0]
                                
                                # 纭繚楂樼偣鏄捐憲楂樹簬涓や釜搴曠偣
                                if (window_close[peak] > window_close[trough1] * 1.03 and
                                    window_close[peak] > window_close[trough2] * 1.03):
                                    
                                    # 妫€鏌ュ舰鎬佹槸鍚﹀凡瀹屾垚 - 浠锋牸绐佺牬棰堢嚎(闃诲姏浣?
                                    if window_close[-1] > window_close[peak]:
                                        df.iloc[i, df.columns.get_loc('pattern_double_bottom')] = 1
            
                # 5. 妫€娴嬩笂鍗囦笁瑙掑舰褰㈡€?(鐪嬫定鎸佺画褰㈡€?
                if len(peaks) >= 2 and len(troughs) >= 2:
                    # 涓婂崌涓夎褰㈡湁骞崇洿鐨勯《閮ㄩ樆鍔涗綅鍜屼笂鍗囩殑搴曢儴鏀拺浣?
                    recent_peaks = sorted(peaks[-3:]) if len(peaks) >= 3 else sorted(peaks[-2:])
                    recent_troughs = sorted(troughs[-3:]) if len(troughs) >= 3 else sorted(troughs[-2:])
                    
                    if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                        # 妫€鏌ラ《閮ㄦ槸鍚︾浉瀵瑰钩鐩?(楂樼偣澶ц嚧鐩稿悓)
                        peak_values = [window_close[p] for p in recent_peaks]
                        if max(peak_values) - min(peak_values) < min(peak_values) * 0.03:
                            # 妫€鏌ュ簳閮ㄦ槸鍚︿笂鍗?(浣庣偣閫愭笎鎶珮)
                            trough_values = [window_close[t] for t in sorted(recent_troughs)]
                            is_ascending = True
                            for j in range(1, len(trough_values)):
                                if trough_values[j] < trough_values[j-1]:
                                    is_ascending = False
                                    break
                                    
                            if is_ascending and trough_values[-1] > trough_values[0] * 1.02:
                                # 妫€鏌ヤ环鏍兼槸鍚︽帴杩戝舰鎬侀《閮?
                                resistance = np.mean(peak_values)
                                if window_close[-1] > resistance * 0.97:
                                    df.iloc[i, df.columns.get_loc('pattern_triangle_ascending')] = 1
                
                # 6. 妫€娴嬩笅闄嶄笁瑙掑舰褰㈡€?(鐪嬭穼鎸佺画褰㈡€?
                if len(peaks) >= 2 and len(troughs) >= 2:
                    # 涓嬮檷涓夎褰㈡湁涓嬮檷鐨勯《閮ㄩ樆鍔涗綅鍜屽钩鐩寸殑搴曢儴鏀拺浣?
                    recent_peaks = sorted(peaks[-3:]) if len(peaks) >= 3 else sorted(peaks[-2:])
                    recent_troughs = sorted(troughs[-3:]) if len(troughs) >= 3 else sorted(troughs[-2:])
                    
                    if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                        # 妫€鏌ュ簳閮ㄦ槸鍚︾浉瀵瑰钩鐩?
                        trough_values = [window_close[t] for t in recent_troughs]
                        if max(trough_values) - min(trough_values) < min(trough_values) * 0.03:
                            # 妫€鏌ラ《閮ㄦ槸鍚︿笅闄?
                            peak_values = [window_close[p] for p in sorted(recent_peaks)]
                            is_descending = True
                            for j in range(1, len(peak_values)):
                                if peak_values[j] > peak_values[j-1]:
                                    is_descending = False
                                    break
                                    
                            if is_descending and peak_values[-1] < peak_values[0] * 0.98:
                                # 妫€鏌ヤ环鏍兼槸鍚︽帴杩戝舰鎬佸簳閮?
                                support = np.mean(trough_values)
                                if window_close[-1] < support * 1.03:
                                    df.iloc[i, df.columns.get_loc('pattern_triangle_descending')] = 1
                
                # 7. 妫€娴嬪绉颁笁瑙掑舰褰㈡€?
                if len(peaks) >= 3 and len(troughs) >= 3:
                    # 瀵圭О涓夎褰㈡湁涓嬮檷鐨勯《閮ㄥ拰涓婂崌鐨勫簳閮?
                    peak_values = [window_close[p] for p in sorted(peaks[-3:])]
                    trough_values = [window_close[t] for t in sorted(troughs[-3:])]
                    
                    # 妫€鏌ラ《閮ㄦ槸鍚︿笅闄?
                    is_tops_descending = True
                    for j in range(1, len(peak_values)):
                        if peak_values[j] > peak_values[j-1]:
                            is_tops_descending = False
                            break
                    
                    # 妫€鏌ュ簳閮ㄦ槸鍚︿笂鍗?
                    is_bottoms_ascending = True
                    for j in range(1, len(trough_values)):
                        if trough_values[j] < trough_values[j-1]:
                            is_bottoms_ascending = False
                            break
                    
                    if is_tops_descending and is_bottoms_ascending:
                        # 瀵圭О涓夎褰㈡敹鏁涳紝璁＄畻鏀舵暃鐐?
                        diff_first = peak_values[0] - trough_values[0]
                        diff_last = peak_values[-1] - trough_values[-1]
                        
                        # 鏀舵暃搴︽鏌?- 鏈宸紓搴旇灏忎簬璧峰宸紓
                        if diff_last < diff_first * 0.7:
                            # 妫€鏌ヤ环鏍兼槸鍚︽帴杩戞敹鏁涚偣
                            midpoint = (peak_values[-1] + trough_values[-1]) / 2
                            if abs(window_close[-1] - midpoint) < diff_last * 0.5:
                                df.iloc[i, df.columns.get_loc('pattern_triangle_symmetrical')] = 1
                
                # 8. 妫€娴嬬湅娑ㄦ棗褰?(Bull Flag) - 涓婃定瓒嬪娍涓殑鐭湡鍥炶皟鏁寸悊
                if len(window_close) >= 20:  # 纭繚鏈夎冻澶熸暟鎹偣
                    # 妫€鏌ュ墠娈垫槸鍚︽湁鏄庢樉涓婃定瓒嬪娍
                    uptrend_window = window_close[:15]
                    uptrend_start = uptrend_window[0]
                    uptrend_end = uptrend_window[-1]
                    
                    if uptrend_end > uptrend_start * 1.05:  # 鑷冲皯5%鐨勪笂娑?
                        # 妫€鏌ユ渶杩戦儴鍒嗘槸鍚︽í鐩樻垨灏忓箙涓嬭穼
                        flag_window = window_close[15:]
                        flag_start = flag_window[0]
                        flag_end = flag_window[-1]
                        
                        # 鏃楀舰搴旇鍦ㄤ笂娑ㄤ箣鍚庢湁灏忓箙鍥炶皟鎴栨í鐩樻暣鐞?
                        if flag_end > flag_start * 0.95 and flag_end < flag_start * 1.02:
                            # 鏈€鍚庝竴涓环鏍煎簲璇ユ帴杩戠獊鐮存暣鐞嗗尯闂翠笂娌?
                            if window_close[-1] > max(flag_window[:-1]) * 0.98:
                                df.iloc[i, df.columns.get_loc('pattern_flag_bullish')] = 1
                
                # 9. 妫€娴嬬湅璺屾棗褰?(Bear Flag) - 涓嬭穼瓒嬪娍涓殑鐭湡鍙嶅脊鏁寸悊
                if len(window_close) >= 20:
                    # 妫€鏌ュ墠娈垫槸鍚︽湁鏄庢樉涓嬭穼瓒嬪娍
                    downtrend_window = window_close[:15]
                    downtrend_start = downtrend_window[0]
                    downtrend_end = downtrend_window[-1]
                    
                    if downtrend_end < downtrend_start * 0.95:  # 鑷冲皯5%鐨勪笅璺?
                        # 妫€鏌ユ渶杩戦儴鍒嗘槸鍚︽í鐩樻垨灏忓箙涓婃定
                        flag_window = window_close[15:]
                        flag_start = flag_window[0]
                        flag_end = flag_window[-1]
                        
                        # 鏃楀舰搴旇鍦ㄤ笅璺屼箣鍚庢湁灏忓箙鍙嶅脊鎴栨í鐩樻暣鐞?
                        if flag_end < flag_start * 1.05 and flag_end > flag_start * 0.98:
                            # 鏈€鍚庝竴涓环鏍煎簲璇ユ帴杩戠獊鐮存暣鐞嗗尯闂翠笅娌?
                            if window_close[-1] < min(flag_window[:-1]) * 1.02:
                                df.iloc[i, df.columns.get_loc('pattern_flag_bearish')] = 1
            
            except Exception as e:
                self.logger.warning(f"妫€娴嬪浘琛ㄥ舰鎬佹椂鍑洪敊: {e}")
                continue
        
        # 缁煎悎褰㈡€佸己搴?
        df['chart_pattern_strength'] = df[pattern_columns].sum(axis=1)
        
        return df
    
    def _find_peaks(self, prices, min_distance=5, prominence=0.01, width=1):
        """
        瀵绘壘浠锋牸搴忓垪涓殑宄板€肩偣
        
        Args:
            prices: 浠锋牸搴忓垪
            min_distance: 宄板€奸棿鏈€灏忚窛绂?
            prominence: 鏈€灏忓嘲鍊煎嚫鏄惧害(鐩稿楂樺害)
            width: 鏈€灏忓嘲鍊煎搴?
            
        Returns:
            list: 宄板€肩储寮曞垪琛?
        """
        try:
            from scipy.signal import find_peaks, peak_prominences

            # 纭繚鏁版嵁瓒冲闀?
            if len(prices) < min_distance * 2:
                return []

            # 璁＄畻骞冲潎浠锋牸鐢ㄤ簬璁剧疆鐩稿prominence
            mean_price = np.mean(prices)
            abs_prominence = prominence * mean_price

            # 鎵惧埌宄板€肩偣
            peaks, properties = find_peaks(
                prices, 
                distance=min_distance,
                prominence=abs_prominence,
                width=width
            )
            
            # 濡傛灉娌℃湁鎵惧埌瓒冲鐨勫嘲鍊硷紝鍙互灏濊瘯闄嶄綆瑕佹眰
            if len(peaks) < 2 and len(prices) >= 20:
                # 闄嶄綆绐佸嚭搴﹁姹?
                peaks, properties = find_peaks(
                    prices, 
                    distance=min(3, min_distance),
                    prominence=abs_prominence * 0.5,
                    width=width
                )
                
            # 鎸夌収绐佸嚭搴﹀宄板€兼帓搴?
            if len(peaks) > 0:
                prominences = peak_prominences(prices, peaks)[0]
                # 鏍规嵁绐佸嚭搴﹀宄板€兼帓搴忥紙闄嶅簭锛?
                sorted_indices = np.argsort(-prominences)
                peaks = peaks[sorted_indices]
                
            return peaks.tolist()
            
        except (ImportError, Exception) as e:
            self.logger.warning(f"宄板€兼娴嬪嚭閿? {e}锛屼娇鐢ㄧ畝鍗曠畻娉曟浛浠?)
            # 绠€鍗曠殑宄板€兼娴嬬畻娉曚綔涓哄閫?
            result = []
            if len(prices) < 7:  # 鏁版嵁澶皯
                return result
                
            for i in range(3, len(prices) - 3):
                # 褰撳墠鐐归珮浜庡叾宸﹀彸3涓偣鍒欒涓哄嘲鍊?
                if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i-3] and 
                    prices[i] > prices[i+1] and prices[i] > prices[i+2] and prices[i] > prices[i+3]):
                    result.append(i)
                    
            return result
        
    def _analyze_price_trend(self, df):
        """
        鍒嗘瀽浠锋牸瓒嬪娍鐘舵€侊細涓婂崌瓒嬪娍銆佷笅闄嶈秼鍔裤€佺洏鏁?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞瓒嬪娍鐘舵€佺殑鏁版嵁
        """
        # 纭繚蹇呰鐨勫垪瀛樺湪
        if 'ma20' not in df.columns or 'ma60' not in df.columns:
            if 'close' in df.columns:
                # 璁＄畻绉诲姩骞冲潎绾?
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['ma60'] = df['close'].rolling(window=60).mean()
            else:
                return df
        
        # 鍒濆鍖栬秼鍔跨姸鎬佸垪
        df['trend_state'] = 'unknown'
        df['trend_strength'] = 0.0
        
        # 鑷冲皯闇€瑕?0涓暟鎹偣鏉ョ‘瀹氳秼鍔?
        if len(df) < 60:
            return df
            
        # 璁＄畻鐭湡鍜岄暱鏈熺Щ鍔ㄥ钩鍧囩嚎鐨勬枩鐜?
        df['ma20_slope'] = df['ma20'].diff(5) / df['ma20'].shift(5)
        df['ma60_slope'] = df['ma60'].diff(10) / df['ma60'].shift(10)
        
        # 瀹氫箟瓒嬪娍鐘舵€?
        for i in range(60, len(df)):
            current_price = df.iloc[i]['close']
            ma20 = df.iloc[i]['ma20']
            ma60 = df.iloc[i]['ma60']
            ma20_slope = df.iloc[i]['ma20_slope']
            ma60_slope = df.iloc[i]['ma60_slope']
            
            # 涓婂崌瓒嬪娍: 浠锋牸鍦ㄥ潎绾夸笂鏂癸紝鐭湡鍧囩嚎鍚戜笂锛岄暱鏈熷潎绾垮悜涓婃垨姘村钩
            if (current_price > ma20 > ma60 and 
                ma20_slope > 0.002 and ma60_slope > -0.0005):
                trend_state = 'uptrend'
                # 寮哄害鐢辨枩鐜囧喅瀹?
                trend_strength = min(1.0, (ma20_slope * 100 + ma60_slope * 50))
                
            # 涓嬮檷瓒嬪娍: 浠锋牸鍦ㄥ潎绾夸笅鏂癸紝鐭湡鍧囩嚎鍚戜笅锛岄暱鏈熷潎绾垮悜涓嬫垨姘村钩
            elif (current_price < ma20 < ma60 and 
                  ma20_slope < -0.002 and ma60_slope < 0.0005):
                trend_state = 'downtrend'
                # 寮哄害鐢辨枩鐜囩粷瀵瑰€煎喅瀹?
                trend_strength = min(1.0, (abs(ma20_slope) * 100 + abs(ma60_slope) * 50))
                
            # 鐩樻暣锛氫环鏍煎湪鐭湡鍜岄暱鏈熷潎绾夸箣闂存尝鍔紝鍧囩嚎鏂滅巼鎺ヨ繎姘村钩
            elif (abs(ma20_slope) < 0.003 and abs(ma60_slope) < 0.002 and
                  abs((current_price / ma20) - 1) < 0.03):
                trend_state = 'consolidation'
                # 鐩樻暣寮哄害鐢变环鏍间笌鍧囩嚎鐨勬帴杩戠▼搴﹀喅瀹?
                trend_strength = 1.0 - abs((current_price / ma20) - 1) * 33
                
            # 绐佺牬锛氫环鏍煎垰鍒氱獊鐮村潎绾匡紝鐭湡鍧囩嚎鎷愬ご鍚戜笂
            elif (current_price > ma20 and 
                  df.iloc[i-1]['close'] <= df.iloc[i-1]['ma20'] and
                  ma20_slope > 0):
                trend_state = 'breakout'
                # 绐佺牬寮哄害鐢变环鏍肩獊鐮村箙搴﹀拰鎴愪氦閲忓喅瀹?
                price_breakout = (current_price / ma20) - 1
                
                # 璁＄畻鎴愪氦閲忓鍔犲€嶆暟锛屾坊鍔犲畨鍏ㄦ鏌ラ伩鍏嶉櫎浠ラ浂
                avg_vol = df.iloc[i-5:i]['volume'].mean()
                if avg_vol > 0:
                    vol_increase = df.iloc[i]['volume'] / avg_vol
                else:
                    vol_increase = 1.0  # 褰撳钩鍧囨垚浜ら噺涓洪浂鏃朵娇鐢ㄩ粯璁ゅ€?
                
                trend_strength = min(1.0, price_breakout * 20 + (vol_increase - 1) * 0.5)
                
            # 鍋囩獊鐮达細浠锋牸绐佺牬鍚庡張鍥炶惤鍒板潎绾夸笅鏂?
            elif (current_price < ma20 and 
                  df.iloc[i-1]['close'] > df.iloc[i-1]['ma20'] and
                  df.iloc[i-2]['close'] > df.iloc[i-2]['ma20'] and
                  ma20_slope < 0):
                trend_state = 'false_breakout'
                # 鍋囩獊鐮村己搴︾敱鍥炶惤骞呭害鍐冲畾
                trend_strength = min(1.0, abs(1 - (current_price / ma20)) * 20)
                
            else:
                trend_state = 'unclear'
                trend_strength = 0.2
                
            df.iloc[i, df.columns.get_loc('trend_state')] = trend_state
            df.iloc[i, df.columns.get_loc('trend_strength')] = trend_strength
            
        # 灏嗚秼鍔跨姸鎬佽浆鎹负鏁板€煎瀷鐗瑰緛锛屼究浜庢ā鍨嬩娇鐢?
        trend_mapping = {
            'uptrend': 2, 
            'breakout': 1, 
            'consolidation': 0, 
            'false_breakout': -1, 
            'downtrend': -2,
            'unclear': 0
        }
        df['trend_state_value'] = df['trend_state'].map(trend_mapping)
        
        return df
        
    def _calculate_risk_reward_ratios(self, df):
        """
        璁＄畻鏀拺浣嶅拰闃诲姏浣嶏紝骞惰浆鎹负鐩堜簭姣?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞鏀拺浣嶅拰闃诲姏浣嶇泩浜忔瘮鐨勬暟鎹?
        """
        # 纭繚蹇呰鐨勫垪瀛樺湪
        if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            return df
            
        # 绐楀彛澶у皬
        window_size = 20
        if len(df) < window_size:
            return df
            
        # 鍒濆鍖栨敮鎾戜綅鍜岄樆鍔涗綅鍒?
        df['support_level'] = np.nan
        df['resistance_level'] = np.nan
        df['risk_reward_ratio'] = np.nan
        
        # 浣跨敤婊戝姩绐楀彛璁＄畻鏀拺浣嶅拰闃诲姏浣?
        for i in range(window_size, len(df)):
            window_close = df.iloc[i-window_size:i]['close'].values
            window_high = df.iloc[i-window_size:i]['high'].values
            window_low = df.iloc[i-window_size:i]['low'].values
            
            # 鎵惧嚭浣庣偣鍜岄珮鐐?
            support_level = np.percentile(window_low, 10)  # 10%鍒嗕綅鏁颁綔涓烘敮鎾戜綅
            resistance_level = np.percentile(window_high, 90)  # 90%鍒嗕綅鏁颁綔涓洪樆鍔涗綅
            
            current_price = df.iloc[i]['close']
            
            # 璁＄畻鍒版敮鎾戜綅鍜岄樆鍔涗綅鐨勮窛绂?
            distance_to_support = (current_price - support_level) / current_price
            distance_to_resistance = (resistance_level - current_price) / current_price
            
            # 纭繚璺濈涓烘鍊间笖涓嶄负闆?
            distance_to_support = max(0.001, distance_to_support)
            distance_to_resistance = max(0.001, distance_to_resistance)
            
            # 璁＄畻鐩堜簭姣?= 娼滃湪鏀剁泭/娼滃湪椋庨櫓
            risk_reward_ratio = distance_to_resistance / distance_to_support
            
            df.iloc[i, df.columns.get_loc('support_level')] = support_level
            df.iloc[i, df.columns.get_loc('resistance_level')] = resistance_level
            df.iloc[i, df.columns.get_loc('risk_reward_ratio')] = risk_reward_ratio
            
        # 娣诲姞鐩稿鏀拺/闃诲姏寮哄害
        df['support_strength'] = 0.0
        df['resistance_strength'] = 0.0
        
        # 璁＄畻鏀拺鍜岄樆鍔涚殑寮哄害
        for i in range(window_size, len(df)):
            if i >= 2 * window_size:
                # 妫€鏌ヨ繃鍘荤殑浠锋牸鏄惁鏇惧湪鏀拺浣嶉檮杩戝弽寮?
                support_level = df.iloc[i]['support_level']
                support_touches = sum(1 for j in range(i-window_size, i) 
                                     if abs(df.iloc[j]['low'] - support_level) / support_level < 0.01)
                df.iloc[i, df.columns.get_loc('support_strength')] = min(1.0, support_touches / 5)
                
                # 妫€鏌ヨ繃鍘荤殑浠锋牸鏄惁鏇惧湪闃诲姏浣嶉檮杩戝洖钀?
                resistance_level = df.iloc[i]['resistance_level']
                resistance_touches = sum(1 for j in range(i-window_size, i) 
                                        if abs(df.iloc[j]['high'] - resistance_level) / resistance_level < 0.01)
                df.iloc[i, df.columns.get_loc('resistance_strength')] = min(1.0, resistance_touches / 5)
        
        return df
        
   
        
    def _generate_state_labels(self, df):
        """
        鐢熸垚缁煎悎鐘舵€佹爣绛撅紝鐢ㄤ簬鍒嗙被鑲＄エ褰撳墠鎵€澶勭殑甯傚満鐘舵€?
        
        Args:
            df: 鍗曞彧鑲＄エ鐨凞ataFrame
            
        Returns:
            DataFrame: 娣诲姞缁煎悎鐘舵€佹爣绛剧殑鏁版嵁
        """
        # 鍒濆鍖栫姸鎬佹爣绛惧垪
        df['stock_state'] = 'neutral'
        
        # 鑷冲皯闇€瑕?0涓暟鎹偣
        if len(df) < 20:
            return df
            
        # 缁煎悎澶氫釜鐗瑰緛纭畾鐘舵€?
        for i in range(20, len(df)):
            # 鑾峰彇褰撳墠琛屾暟鎹?
            row = df.iloc[i]
            
            # 妫€鏌ユ槸鍚︽湁瓒嬪娍鐘舵€?
            if 'trend_state' in df.columns:
                trend = row['trend_state']
            else:
                trend = 'unknown'
                
            # 妫€鏌绾垮舰鎬?
            bullish_patterns = sum([
                row.get('pattern_hammer', 0),
                row.get('pattern_engulfing_bullish', 0),
                row.get('pattern_morning_star', 0),
                row.get('pattern_three_white_soldiers', 0)
            ])
            
            bearish_patterns = sum([
                row.get('pattern_hanging_man', 0),
                row.get('pattern_engulfing_bearish', 0),
                row.get('pattern_evening_star', 0),
                row.get('pattern_three_black_crows', 0)
            ])
            
            # 妫€鏌ラ闄╂敹鐩婃瘮
            risk_reward = row.get('risk_reward_ratio', 1.0)
            
            # 妫€鏌SI瓒呬拱瓒呭崠
            rsi_value = row.get('rsi12', 50)
            
            # 妫€鏌ユ垚浜ら噺寮傚父
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # 缁撳悎鍚勭鐗瑰緛纭畾鐘舵€?
            if trend == 'uptrend' and bullish_patterns >= 1 and risk_reward > 2.0:
                state = 'strong_buy'
            elif trend == 'breakout' and volume_ratio > 1.5:
                state = 'breakout_buy'
            elif trend == 'uptrend' or (bullish_patterns >= 1 and rsi_value < 70):
                state = 'buy'
            elif trend == 'downtrend' and bearish_patterns >= 1 and risk_reward < 0.5:
                state = 'strong_sell'
            elif trend == 'false_breakout' and volume_ratio > 1.5:
                state = 'false_breakout_sell'
            elif trend == 'downtrend' or (bearish_patterns >= 1 and rsi_value > 30):
                state = 'sell'
            elif trend == 'consolidation' and 0.8 < risk_reward < 1.2:
                state = 'consolidation'
            elif rsi_value < 30 and risk_reward > 1.5:
                state = 'oversold'
            elif rsi_value > 70 and risk_reward < 0.7:
                state = 'overbought'
            else:
                state = 'neutral'
                
            df.iloc[i, df.columns.get_loc('stock_state')] = state
            
        # 灏嗙姸鎬佽浆鎹负鏁板€硷紝渚夸簬妯″瀷浣跨敤
        state_mapping = {
            'strong_buy': 2, 
            'breakout_buy': 1.5,
            'buy': 1, 
            'neutral': 0, 
            'sell': -1, 
            'false_breakout_sell': -1.5,
            'strong_sell': -2,
            'consolidation': 0,
            'oversold': 0.5,
            'overbought': -0.5
        }
        df['stock_state_value'] = df['stock_state'].map(state_mapping)
        
        return df
            
    def _load_market_status_by_date(self, date):
        """
        鍔犺浇鎸囧畾鏃ユ湡鐨勫競鍦虹姸鎬佹暟鎹?
        
        Args:
            date: 鏃ユ湡锛屾牸寮忎负YYYYMMDD
            
        Returns:
            dict: 甯傚満鐘舵€佹暟鎹?
        """
        try:
            # 灏嗘棩鏈熻浆鎹负瀛楃涓叉牸寮?
            date_str = date
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y%m%d')
                
            # 棣栧厛灏濊瘯浠巁load_all_market_status鑾峰彇鐨勬暟鎹腑鏌ユ壘
            all_market_status = self._load_all_market_status()
            if all_market_status and date_str in all_market_status:
                return all_market_status[date_str]
            return None    
        except Exception as e:
            self.logger.warning(f"鍔犺浇鏃ユ湡 {date} 鐨勫競鍦虹姸鎬佹暟鎹椂鍑洪敊: {e}")
            return None

    def _detect_stock_change_points(self, stock_features_df):
        """
        妫€娴嬭偂绁ㄤ环鏍煎叧閿彉鍖栫偣
        
        Args:
            stock_features_df: 鑲＄エ鐗瑰緛DataFrame
            
        Returns:
            dict: 鍙樺寲鐐规娴嬬粨鏋滐紝鏍煎紡涓?{stock_code: {'change_points': [...], 'returns': [...], 'dates': [...]}}
        """
        # 妫€鏌ョ紦瀛?
        # 鍒涘缓缂撳瓨鐩綍
        cache_dir = self.model_dir / "features_cache"
        ensure_dir(cache_dir)
        
        # 鐢熸垚缂撳瓨鏂囦欢鍚?
        try:
            date_range = f"{pd.to_datetime(stock_features_df['date_str']).min().strftime('%Y%m%d')}_{pd.to_datetime(stock_features_df['date_str']).max().strftime('%Y%m%d')}"
            num_stocks = len(stock_features_df['stock_code'].unique())
            cache_file = cache_dir / f"change_points_{num_stocks}stocks_{date_range}.pkl"
            
            # 濡傛灉缂撳瓨瀛樺湪锛岀洿鎺ュ姞杞藉苟杩斿洖
            if cache_file.exists():
                self.logger.info(f"浠庣紦瀛樺姞杞藉彉鍖栫偣妫€娴嬬粨鏋? {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"鍔犺浇缂撳瓨澶辫触: {e}锛屽皢閲嶆柊璁＄畻鍙樺寲鐐?)
        except Exception as e:
            self.logger.warning(f"鐢熸垚缂撳瓨閿け璐? {e}锛屽皢涓嶄娇鐢ㄧ紦瀛?)
        
        self.logger.info("寮€濮嬫娴嬭偂绁ㄤ环鏍煎叧閿彉鍖栫偣...")
        
        # 浠呬娇鐢╢uture_return_2d浣滀负鍙樺寲鐐规娴嬬殑鏍稿績鎸囨爣
        # 鍏朵粬鎸囨爣浠呬綔涓哄弬鑰冿紝浣嗕笉鐩存帴鐢ㄤ簬鍙樺寲鐐规娴?
        indicators = {'future_return_2d': 1.0}  # 鏈€鏍稿績鎸囨爣锛屽敮涓€鏉冮噸
        
        # 妫€鏌ユ槸鍚︽湁future_return_2d鍒?
        if 'future_return_2d' not in stock_features_df.columns:
            self.logger.error(f"鑲＄エ鐗瑰緛鏁版嵁涓己灏戝繀瑕佺殑future_return_2d鍒楋紝鏃犳硶杩涜鍙樺寲鐐规娴?)
            self.logger.info(f"鍙敤鐨勫垪鏈? {stock_features_df.columns.tolist()}")
            
            # 妫€鏌ユ槸鍚﹂渶瑕侀噸鏂拌绠梖uture_return_2d
            if 'close' in stock_features_df.columns and 'date_str' in stock_features_df.columns:
                self.logger.info("灏濊瘯閲嶆柊璁＄畻future_return_2d鍒?..")
                
                # 鎸夎偂绁ㄥ垎缁勫鐞?
                grouped = stock_features_df.groupby('stock_code')
                all_processed = []
                
                for stock_code, stock_df in grouped:
                    try:
                        # 璁＄畻future_return_2d
                        processed_df = self._calculate_future_returns(stock_df)
                        all_processed.append(processed_df)
                    except Exception as e:
                        self.logger.error(f"閲嶆柊璁＄畻鑲＄エ {stock_code} 鐨刦uture_return_2d鏃跺嚭閿? {e}")
                        all_processed.append(stock_df)  # 淇濈暀鍘熷鏁版嵁
                
                # 鍚堝苟缁撴灉
                if all_processed:
                    stock_features_df = pd.concat(all_processed, ignore_index=False)
                    self.logger.info(f"閲嶆柊璁＄畻future_return_2d瀹屾垚锛岀幇鍦ㄦ湁鏁堝€兼瘮渚? {stock_features_df['future_return_2d'].notna().sum() / len(stock_features_df):.2f}")
                else:
                    self.logger.error("閲嶆柊璁＄畻future_return_2d澶辫触")
                    return {}
            else:
                self.logger.error("缂哄皯璁＄畻future_return_2d鎵€闇€鐨勫繀瑕佸垪锛屾棤娉曠户缁彉鍖栫偣妫€娴?)
                return {}
        
        # 澶勭悊涓嶅悓鑲＄エ
        results = {}
        stock_codes = stock_features_df['stock_code'].unique()
        
        self.logger.info(f"寮€濮嬪鐞?{len(stock_codes)} 鍙偂绁ㄧ殑鍙樺寲鐐规娴?..")
        
        # 鎸夎偂绁ㄤ唬鐮佸垎缁勫鐞?
        for i, stock_code in enumerate(stock_codes):
            if i % 100 == 0:
                self.logger.info(f"宸插鐞?{i}/{len(stock_codes)} 鍙偂绁?)
                
            stock_df = stock_features_df[stock_features_df['stock_code'] == stock_code].copy()
            if len(stock_df) < 30:  # 纭繚鏈夎冻澶熸暟鎹?
                continue
                
            # 纭繚鑲＄エ鏁版嵁涓湁future_return_2d鍒?
            if 'future_return_2d' not in stock_df.columns:
                self.logger.warning(f"鑲＄エ {stock_code} 缂哄皯future_return_2d鍒楋紝鏃犳硶杩涜鍙樺寲鐐规娴?)
                continue
                
            # 妫€鏌uture_return_2d鍒楁槸鍚︽湁瓒冲鐨勬湁鏁堝€?
            valid_ratio = stock_df['future_return_2d'].notna().sum() / len(stock_df)
            if valid_ratio < 0.3:  # 鑷冲皯30%鐨勬暟鎹湁鏁?
                self.logger.warning(f"鑲＄エ {stock_code} 鐨刦uture_return_2d鏈夋晥鍊兼瘮渚嬭繃浣? {valid_ratio:.2f}锛岃烦杩?)
                continue
                
            # 璋冪敤鍗曞彧鑲＄エ鐨勫彉鍖栫偣妫€娴嬫柟娉?
            result = self._detect_single_stock_change_points(stock_code, stock_df, indicators)
            
            # 淇濆瓨鏈夋晥缁撴灉
            if result and 'change_points' in result and len(result['change_points']) > 0:
                results[stock_code] = result
        
        successful_count = len(results)
        total_count = len(stock_codes)
        success_rate = successful_count / total_count * 100 if total_count > 0 else 0
        
        self.logger.info(f"鍙樺寲鐐规娴嬪畬鎴愶紝鎴愬姛鐜? {success_rate:.1f}% ({successful_count}/{total_count})")
        
        # 璁＄畻鎵€鏈夎偂绁ㄥ钩鍧囧彉鍖栫偣鏁伴噺
        if results:
            total_points = sum(len(r['change_points']) for r in results.values() if 'change_points' in r)
            avg_points = total_points / len(results)
            self.logger.info(f"骞冲潎姣忎釜鑲＄エ鍙樺寲鐐规暟: {avg_points:.2f}")
            
        # 淇濆瓨缁撴灉鍒扮紦瀛?
        try:
            if 'cache_file' in locals():
                self.logger.info(f"淇濆瓨鍙樺寲鐐规娴嬬粨鏋滃埌缂撳瓨: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
        except Exception as e:
            self.logger.warning(f"淇濆瓨缂撳瓨澶辫触: {e}")
            
        return results

    def calculate_stock_sector_relative_features(self, stock_features_df, sector_features_df, change_points=None, is_training=False, use_cache=True):
        """
        璁＄畻涓偂鐩稿浜庢澘鍧楃殑鐩稿鐗瑰緛锛屽彧澶勭悊鍙樺寲鐐瑰搴旂殑鏃ユ湡
        
        Args:
            stock_features_df: 涓偂鐗瑰緛DataFrame
            sector_features_df: 鏉垮潡鐗瑰緛DataFrame
            change_points: 鍙樺寲鐐规娴嬬粨鏋滐紝鏍煎紡涓?{stock_code: {'change_points': [...], 'dates': [...]}}
            is_training: 鏄惁涓鸿缁冩暟鎹紝鐢ㄤ簬鍖哄垎缂撳瓨鏂囦欢
            use_cache: 鏄惁浣跨敤缂撳瓨锛岄粯璁や负True
            
        Returns:
            tuple: (娣诲姞浜嗙浉瀵圭壒寰佺殑DataFrame, 鑲＄エ鏉垮潡鍏宠仈缁撴灉DataFrame)
        """
        self.logger.info("寮€濮嬭绠椾釜鑲＄浉瀵逛簬鏉垮潡鐨勭壒寰?..")
        
        if stock_features_df.empty or sector_features_df.empty:
            self.logger.warning("鐗瑰緛鏁版嵁涓虹┖锛岃繑鍥炵┖DataFrame")
            return stock_features_df, pd.DataFrame()
        
        # 妫€鏌ョ紦瀛?
        cache_file = None
        if use_cache:
            # 鍒涘缓缂撳瓨鐩綍
            cache_dir = self.model_dir / "features_cache"
            ensure_dir(cache_dir)
            
            # 鐢熸垚缂撳瓨鏂囦欢鍚?
            data_type = "train" if is_training else "predict"
            
            # 浣跨敤鏃ユ湡鑼冨洿鍜岃偂绁ㄦ暟閲忎綔涓虹紦瀛橀敭鐨勪竴閮ㄥ垎
            try:
                date_range = f"{pd.to_datetime(stock_features_df['date_str']).min().strftime('%Y%m%d')}_{pd.to_datetime(stock_features_df['date_str']).max().strftime('%Y%m%d')}"
                num_stocks = len(stock_features_df['stock_code'].unique())
                cache_file = cache_dir / f"relative_features_{data_type}_{num_stocks}stocks_{date_range}.pkl"
                
                # 濡傛灉缂撳瓨瀛樺湪锛岀洿鎺ュ姞杞藉苟杩斿洖
                if cache_file.exists():
                    self.logger.info(f"浠庣紦瀛樺姞杞界浉瀵圭壒寰? {cache_file}")
                    try:
                        # 鍔犺浇鐗瑰緛缂撳瓨
                        with open(cache_file, 'rb') as f:
                            result_df = pickle.load(f)
                            
                        # 灏濊瘯鍔犺浇鍏崇郴鏁版嵁缂撳瓨
                        relations_cache_file = cache_file.parent / f"relations_{cache_file.name}"
                        relations_df = pd.DataFrame()
                        
                        if relations_cache_file.exists():
                            self.logger.info(f"浠庣紦瀛樺姞杞芥澘鍧楀叧鑱旂粨鏋? {relations_cache_file}")
                            try:
                                with open(relations_cache_file, 'rb') as f:
                                    relations_df = pickle.load(f)
                                self.logger.info(f"鎴愬姛鍔犺浇鏉垮潡鍏宠仈缁撴灉缂撳瓨锛屽寘鍚?{len(relations_df)} 鏉¤褰?)
                            except Exception as e:
                                self.logger.warning(f"鍔犺浇鏉垮潡鍏宠仈缁撴灉缂撳瓨澶辫触: {e}锛屽皢浣跨敤绌篋ataFrame")
                                relations_df = pd.DataFrame()
                        
                        return result_df, relations_df
                    except Exception as e:
                        self.logger.warning(f"鍔犺浇缂撳瓨澶辫触: {e}锛屽皢閲嶆柊璁＄畻鐗瑰緛")
            except Exception as e:
                self.logger.warning(f"鐢熸垚缂撳瓨閿け璐? {e}锛屽皢涓嶄娇鐢ㄧ紦瀛?)
        
        # 鎬ц兘浼樺寲锛氬鍒舵暟鎹墠鍏堝垱寤虹储寮曪紝閬垮厤鍙嶅鏌ヨ
        self.logger.info("鍒涘缓楂樻晥绱㈠紩浠ュ姞閫熷鐞?..")
        
        # 澶嶅埗鏁版嵁锛岄伩鍏嶄慨鏀瑰師濮嬫暟鎹?
        result_df = stock_features_df.copy()
        
        # 棰勫鐞嗚偂绁ㄩ緳澶寸壒寰?
        result_df = self._prepare_stock_leadership_features(result_df)
        
        # 鑾峰彇鑲＄エ鏉垮潡鏄犲皠鍑芥暟
        get_stock_sectors_func = self._load_stock_sector_mapping()
        
        # 鍔犺浇涓€娆℃暟鎹紝渚涙墍鏈夎偂绁ㄤ娇鐢?
        data = None
        try:
            from to.stock_sectors_mapping import load_data
            data = load_data()
            self.logger.info("鎴愬姛鍔犺浇鑲＄エ鏉垮潡鏄犲皠鏁版嵁")
        except ImportError:
            try:
                from stock_sectors_mapping import load_data
                data = load_data()
                self.logger.info("鎴愬姛鍔犺浇鑲＄エ鏉垮潡鏄犲皠鏁版嵁")
            except ImportError:
                self.logger.warning("鏃犳硶瀵煎叆load_data鍑芥暟锛岃偂绁ㄦ澘鍧楁槧灏勫彲鑳戒笉瀹屾暣")
            except Exception as e:
                self.logger.warning(f"鍔犺浇stock_sectors_mapping鏁版嵁鏃跺嚭閿? {str(e)}")
        except Exception as e:
            self.logger.warning(f"鍔犺浇to.stock_sectors_mapping鏁版嵁鏃跺嚭閿? {str(e)}")
        
        # 鍒濆鍖栫浉瀵圭壒寰佸垪
        relative_cols = [
            'most_relevant_sector',      # 鏈€鐩稿叧鏉垮潡
            'sector_correlation_score',  # 涓庢澘鍧楃浉鍏虫€у緱鍒?
            'stock_sector_rank',         # 鑲＄エ鍦ㄦ澘鍧椾腑鐨勬帓鍚?
            'is_sector_leader',          # 鏄惁涓烘澘鍧楅緳澶?
            'relative_return',           # 涓偂鐩稿鏉垮潡鏀剁泭鐜?
            'relative_turnover',         # 涓偂鐩稿鏉垮潡鎹㈡墜鐜?
            'relative_volume',           # 涓偂鐩稿鏉垮潡鎴愪氦閲?
            'relative_momentum',         # 涓偂鐩稿鏉垮潡鍔ㄩ噺
            'relative_volatility',       # 涓偂鐩稿鏉垮潡娉㈠姩鐜?
            'relative_strength_index',   # 鐩稿寮哄急鎸囨暟
            'sector_deviation'           # 涓偂瀵规澘鍧楃殑鑳岀绋嬪害
        ]
        
        # 鍒濆鍖栧瓧绗︿覆鍒椾负None锛屾暟鍊煎垪涓?.0
        for col in relative_cols:
            if col == 'most_relevant_sector':
                result_df[col] = None
            elif col == 'is_sector_leader':
                result_df[col] = False
            else:
                result_df[col] = 0.0
                
        # 鎺掗櫎鐨勯潪琛屼笟姒傚康鏉垮潡鍒楄〃
        excluded_sectors = [
            'TGN涓瘉500鎴愪唤鑲?, 'TGN铻嶈祫铻嶅埜', 'TGN娣辫偂閫?, 
            'TGN鏍囨櫘閬撶惣鏂疉鑲?, 'TGNMSCI姒傚康', 'TGN娌繁300鏍锋湰鑲?
        ]
        
        # 鍒涘缓鑲＄エ鍒版澘鍧楃殑鏄犲皠缂撳瓨锛屼竴娆℃€ц幏鍙栨墍鏈夋槧灏?
        self.logger.info("涓€娆℃€ф瀯寤烘墍鏈夎偂绁ㄧ殑鏉垮潡鏄犲皠鍏崇郴...")
        stock_codes = result_df['stock_code'].unique()
        stock_sector_mapping = {}
        
        # 浣跨敤澶氳繘绋嬪姞閫熻偂绁ㄦ澘鍧楁槧灏勬瀯寤?
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        
        def get_stock_sectors_wrapper(stock_code):
            sectors = get_stock_sectors_func(stock_code=stock_code, data=data)
            if sectors:
                return stock_code, [s for s in sectors if s not in excluded_sectors]
            return stock_code, []
        
        # 浣跨敤绾跨▼姹犲苟琛屽鐞嗚偂绁ㄥ埌鏉垮潡鐨勬槧灏?
        with ThreadPoolExecutor(max_workers=10) as executor:
            for stock_code, sectors in executor.map(get_stock_sectors_wrapper, stock_codes):
                stock_sector_mapping[stock_code] = sectors
                
        # 纭畾闇€瑕佸鐞嗙殑鑲＄エ-鏃ユ湡缁勫悎
        process_combinations = []
        
        # 淇闂1: 姝ｇ‘妫€鏌ュ彉鍖栫偣鏁版嵁鏄惁鏈夋晥
        has_valid_change_points = False
        if change_points:
            # 妫€鏌ユ槸鍚︽湁浠讳綍鏈夋晥鐨勫彉鍖栫偣
            for stock_code, cp_data in change_points.items():
                if cp_data and isinstance(cp_data, dict) and 'dates' in cp_data:
                    dates = cp_data['dates']
                    if isinstance(dates, list) and dates:
                        has_valid_change_points = True
                        for date_str in dates:
                            process_combinations.append((stock_code, date_str))
                    elif isinstance(dates, np.ndarray) and dates.size > 0:
                        has_valid_change_points = True
                        for date_str in dates:
                            process_combinations.append((stock_code, date_str))
            
            if has_valid_change_points:
                self.logger.info(f"灏嗗鐞?{len(process_combinations)} 涓彉鍖栫偣瀵瑰簲鐨勮偂绁?鏃ユ湡缁勫悎")
            else:
                self.logger.warning("鏈壘鍒版湁鏁堢殑鍙樺寲鐐规暟鎹紝灏嗗鐞嗘墍鏈夎偂绁?鏃ユ湡缁勫悎")
                has_valid_change_points = False
        
        if not has_valid_change_points:
            # 濡傛灉娌℃湁鎻愪緵鍙樺寲鐐癸紝鍒欏鐞嗘墍鏈夌粍鍚?
            self.logger.warning("灏嗗鐞嗘墍鏈夎偂绁?鏃ユ湡缁勫悎")
            stock_groups = result_df.groupby(['stock_code', 'date_str'])
            process_combinations = [(stock_code, date_str) for (stock_code, date_str) in stock_groups.groups.keys()]
        
        # 鍒涘缓鎬ц兘浼樺寲鐨勭储寮?
        result_df_idx = result_df.set_index(['stock_code', 'date_str'])
        sector_features_df_idx = sector_features_df.set_index(['sector_code', 'date'])
        
        # 鎵瑰鐞嗭紝姣忔壒澶勭悊1000涓粍鍚?
        batch_size = 1000
        total_groups = len(process_combinations)
        processed = 0
        
        # 鍒涘缓缁撴灉瀛楀吀
        results = {}
        
        # 娣诲姞杩涘害鏃ュ織 - 淇闂3
        self.logger.info(f"寮€濮嬪鐞?{total_groups} 涓偂绁?鏃ユ湡缁勫悎锛屾瘡鎵规 {batch_size} 涓?..")
        start_time = time.time()
        
        # 鎸夋壒娆″鐞?
        for batch_start in range(0, total_groups, batch_size):
            batch_end = min(batch_start + batch_size, total_groups)
            batch = process_combinations[batch_start:batch_end]
            
            batch_start_time = time.time()
            self.logger.info(f"寮€濮嬪鐞嗙 {batch_start//batch_size + 1} 鎵癸紝鍏?{len(batch)} 涓粍鍚?..")
            
            # 浣跨敤绾跨▼姹犲苟琛屽鐞嗘壒娆?
            with ThreadPoolExecutor(max_workers=min(20, os.cpu_count() * 5)) as executor:
                # 鎻愪氦鎵€鏈変换鍔?
                futures_dict = {}
                for stock_code, date_str in batch:
                    future = executor.submit(
                        self._process_single_stock_date, 
                        stock_code, 
                        date_str, 
                        result_df_idx, 
                        stock_sector_mapping, 
                        sector_features_df_idx
                    )
                    futures_dict[future] = (stock_code, date_str)
                
                # 鏀堕泦缁撴灉
                batch_success = 0
                batch_errors = 0
                for future in as_completed(futures_dict):
                    try:
                        result = future.result()
                        if result:
                            key = (result['stock_code'], result['date'])
                            results[key] = result
                            batch_success += 1
                    except Exception as e:
                        stock_code, date_str = futures_dict[future]
                        self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鍦?{date_str} 鐨勭壒寰佹椂鍑洪敊: {str(e)}")
                        batch_errors += 1
            
            # 鏇存柊澶勭悊杩涘害
            processed += len(batch)
            batch_time = time.time() - batch_start_time
            self.logger.info(f"绗?{batch_start//batch_size + 1} 鎵瑰鐞嗗畬鎴愶細鎴愬姛 {batch_success}锛屽け璐?{batch_errors}锛岀敤鏃?{batch_time:.2f} 绉?)
            self.logger.info(f"鎬昏繘搴? 宸插鐞?{processed}/{total_groups} 涓偂绁?鏃ユ湡缁勫悎 ({processed/total_groups:.1%})")
        
        total_time = time.time() - start_time
        self.logger.info(f"鎵€鏈夋壒娆″鐞嗗畬鎴愶紝鎬荤敤鏃?{total_time:.2f} 绉掞紝骞冲潎姣忎釜缁勫悎 {total_time/total_groups:.4f} 绉?)
        
        # 灏嗙粨鏋滃瓧鍏歌浆鎹负DataFrame鏇存柊鍘熷鏁版嵁
        if results:
            self.logger.info(f"鍏辨壘鍒?{len(results)} 涓湁鏁堢殑鑲＄エ-鏉垮潡鍏宠仈")
            try:
                results_df = pd.DataFrame.from_dict(results, orient='index')
                
                # 閲嶅缓绱㈠紩渚夸簬鍚堝苟
                results_df = results_df.reset_index(drop=True)
                
                self.logger.info(f"鏇存柊绱㈠紩瀹屾垚")
                # 鍒涘缓涓€涓槧灏勭敤浜庢洿鏂板師濮婦ataFrame
                update_map = {
                    (row['stock_code'], row['date']): {
                        'most_relevant_sector': row['best_sector'],
                        'sector_correlation_score': row['best_score'],
                        'is_sector_leader': row.get('best_leadership', 0) > 0.5
                    } for _, row in results_df.iterrows()
                }
                
                # 涓嶅啀灏唘pdate_map鐨勬暟鎹悎骞跺埌result_df涓紝閬垮厤鑰楁椂鎿嶄綔
                # 灏唘pdate_map淇濆瓨鍒皉elations_df涓紝渚涘悗缁柟娉曠洿鎺ヤ娇鐢?
                self.logger.info("璺宠繃鑰楁椂鐨凞ataFrame鏇存柊鎿嶄綔锛岀洿鎺ヤ娇鐢╮elations_df杩涜鏌ヨ")
                
                # 涓轰簡淇濇寔涓庡師濮嬫暟鎹浉浼肩殑缁撴瀯锛屾垜浠皢update_map杞崲涓篋ataFrame
                relations_rows = []
                for (stock_code, date_str), values in update_map.items():
                    row = {'stock_code': stock_code, 'date': date_str}
                    row.update(values)
                    relations_rows.append(row)
                
                # 鍒涘缓relations_df
                relations_df = pd.DataFrame(relations_rows)
                self.logger.info(f"鍒涘缓鏉垮潡鍏宠仈DataFrame锛屽寘鍚?{len(relations_df)} 鏉¤褰?)
            except Exception as e:
                self.logger.error(f"澶勭悊缁撴灉鏁版嵁鏃跺嚭閿? {str(e)}")
        else:
            self.logger.warning("娌℃湁鎵惧埌鏈夋晥鐨勮偂绁?鏉垮潡鍏宠仈鏁版嵁")
        
        # 淇濆瓨璁＄畻缁撴灉鍒扮紦瀛?
        if use_cache and cache_file:
            try:
                self.logger.info(f"淇濆瓨鐩稿鐗瑰緛鍒扮紦瀛樻枃浠? {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(result_df, f)
            except Exception as e:
                self.logger.warning(f"淇濆瓨缂撳瓨澶辫触: {e}")
        
                # 濡傛灉鍓嶉潰鐨勪唬鐮佹病鏈夊垱寤簉elations_df锛堟瘮濡傚湪缁撴灉涓虹┖鐨勬儏鍐典笅锛夛紝鍒欏垱寤轰竴涓┖鐨凞ataFrame
        if 'relations_df' not in locals():
            self.logger.info("娌℃湁鎵惧埌鏈夋晥鐨勮偂绁ㄦ澘鍧楀叧鑱旀暟鎹紝鍒涘缓绌虹殑relations_df")
            relations_df = pd.DataFrame()
        
        # 鍒涘缓鍏宠仈缁撴灉缂撳瓨鏂囦欢
        if use_cache and cache_file:
            try:
                relations_cache_file = cache_file.parent / f"relations_{cache_file.name}"
                self.logger.info(f"淇濆瓨鏉垮潡鍏宠仈缁撴灉鍒扮紦瀛樻枃浠? {relations_cache_file}")
                with open(relations_cache_file, 'wb') as f:
                    pickle.dump(relations_df, f)
            except Exception as e:
                self.logger.warning(f"淇濆瓨鍏宠仈缁撴灉缂撳瓨澶辫触: {e}")
        
        return result_df, relations_df

    def _load_stock_sector_mapping(self):
        """
        鍔犺浇鑲＄エ涓庢澘鍧楃殑鏄犲皠鍏崇郴鍑芥暟
        
        Returns:
            function: get_stock_sectors鍑芥暟寮曠敤锛岀敤浜庤幏鍙栬偂绁ㄦ澘鍧楁槧灏?
        """
        try:
            # 灏濊瘯浠巗tock_sectors_mapping妯″潡鑾峰彇
            try:
                # 灏嗗綋鍓嶅伐浣滅洰褰曟坊鍔犲埌sys.path
                sys.path.append(os.getcwd())
                # 灏濊瘯瀵煎叆
                try:
                    from stock_sectors_mapping import get_stock_sectors
                    self.logger.info("鎴愬姛瀵煎叆stock_sectors_mapping.get_stock_sectors鍑芥暟")
                    return get_stock_sectors
                except ImportError:
                    # 濡傛灉瀵煎叆澶辫触锛屽皾璇曚粠to鐩綍瀵煎叆
                    try:
                        sys.path.append(str(get_project_root()))
                        from to.stock_sectors_mapping import get_stock_sectors
                        self.logger.info("鎴愬姛瀵煎叆to.stock_sectors_mapping.get_stock_sectors鍑芥暟")
                        return get_stock_sectors
                    except ImportError:
                        self.logger.warning("鏃犳硶瀵煎叆get_stock_sectors鍑芥暟")
            except Exception as e:
                self.logger.error(f"灏濊瘯瀵煎叆get_stock_sectors鍑芥暟鏃跺嚭閿? {e}")
            
            # 濡傛灉閮藉け璐ヤ簡锛岃繑鍥炰竴涓粯璁ゅ嚱鏁帮紝璇ュ嚱鏁板缁堣繑鍥炵┖瀛楀吀
            self.logger.warning("鏈兘瀵煎叆get_stock_sectors鍑芥暟锛屽皢浣跨敤榛樿绌烘槧灏勫嚱鏁?)
            return lambda *args, **kwargs: {}
            
        except Exception as e:
            self.logger.error(f"鍔犺浇鑲＄エ鏉垮潡鏄犲皠鍑芥暟鏃跺嚭閿? {e}")
            return lambda *args, **kwargs: {}

    def _add_board_structure_features(self, features, current_day, sector_code, board_structure=None):
        """
        娣诲姞鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰侊紝鍊熼壌sector_stock_analysis涓殑鍒嗘瀽閫昏緫
        
        Args:
            features: 鐗瑰緛瀛楀吀
            current_day: 褰撴棩鏁版嵁
            sector_code: 鏉垮潡浠ｇ爜
            board_structure: 娑ㄨ穼鍋滄澘缁撴瀯鐗瑰緛鏁版嵁
            
        Returns:
            鏇存柊鍚庣殑鐗瑰緛瀛楀吀
        """
        # 娉ㄩ噴: 鏈柟娉曟槸鐗瑰緛鎻愬彇鐨勯噸瑕佺粍鎴愰儴鍒嗭紝閰嶅悎鏁翠綋鐨勭浉浼煎害鍖归厤鏂规硶锛?
        # 鎴戜滑閫氳繃鎻愬彇鍘嗗彶鏁版嵁涓殑鍚勭鐗瑰緛锛堝寘鎷定璺屽仠鏉跨粨鏋勭壒寰侊級锛?
        # 鍦ㄩ娴嬫椂閫氳繃璁＄畻褰撳墠甯傚満鐗瑰緛涓庡巻鍙茬壒寰佺殑鐩镐技搴︽潵璇嗗埆娼滃湪鐨勪氦鏄撴満浼氥€?
        # 杩欑鍩轰簬鐩镐技搴︾殑妯″紡璇嗗埆鏂规硶鍦ㄩ噺鍖栦氦鏄撲腑琚箍娉涘簲鐢紝
        # 鏃犻渶澶嶆潅鐨勬満鍣ㄥ涔犺缁冭繃绋嬶紝浣嗚兘澶熸湁鏁堟崟鎹夊競鍦烘ā寮忋€?
        try:
            # 鍙湁褰揵oard_structure涓嶄负None鏃舵墠娣诲姞鏉垮潡娑ㄨ穼鍋滅粨鏋勭壒寰?
            if board_structure is not None:
                # 1. 娑ㄨ穼鍋滅粺璁＄壒寰?
                for field in ["limit_up_count", "limit_down_count", "near_limit_up_count", 
                             "near_limit_down_count", "one_word_limit_up_count", 
                             "broken_limit_up_count", "limit_open_count"]:
                    if field in board_structure:
                        features[field] = board_structure[field]
                
                # 2. 杩炴澘鐗瑰緛
                if "max_consecutive_limit_up" in board_structure:
                    features["max_consecutive_limit_up"] = board_structure["max_consecutive_limit_up"]
                
                # 3. 婧环鐗瑰緛
                for field in ["yesterday_limit_up_premium", "yesterday_max_limit_up_premium"]:
                    if field in board_structure:
                        features[field] = board_structure[field]
                
                # 4. 甯傚€煎垎甯冪壒寰?
                if "market_cap_distribution" in board_structure and isinstance(board_structure["market_cap_distribution"], dict):
                    for size, count in board_structure["market_cap_distribution"].items():
                        features[f"market_cap_{size}"] = count
                
                # 5. 娑ㄥ仠鏉垮競鍊煎垎甯?
                if "limit_board_distribution" in board_structure and isinstance(board_structure["limit_board_distribution"], dict):
                    for size, count in board_structure["limit_board_distribution"].items():
                        features[f"limit_board_{size}"] = count
                
                # 6. 璁＄畻娑ㄥ仠缁撴瀯鎸囨爣
                if "stock_count" in board_structure and board_structure["stock_count"] > 0:
                    # 娑ㄥ仠鍗犳瘮
                    if "limit_up_count" in board_structure:
                        features["limit_up_ratio"] = board_structure["limit_up_count"] / board_structure["stock_count"]
                    
                    # 璺屽仠鍗犳瘮
                    if "limit_down_count" in board_structure:
                        features["limit_down_ratio"] = board_structure["limit_down_count"] / board_structure["stock_count"]
                    
                    # 涓€瀛楁澘鍗犳瘮
                    if "one_word_limit_up_count" in board_structure:
                        features["one_word_limit_ratio"] = board_structure["one_word_limit_up_count"] / max(1, board_structure["limit_up_count"])
                    
                    # 鐐告澘鐜?
                    if "broken_limit_up_count" in board_structure and "limit_up_count" in board_structure:
                        total_limits = board_structure["limit_up_count"] + board_structure["broken_limit_up_count"]
                        if total_limits > 0:
                            features["broken_ratio"] = board_structure["broken_limit_up_count"] / total_limits
                
                # 7. 娑ㄥ仠绫诲瀷鍒嗗竷
                if "limit_up_types" in board_structure and isinstance(board_structure["limit_up_types"], dict):
                    for type_key, count in board_structure["limit_up_types"].items():
                        features[f"limit_type_{type_key}"] = count
                        
                        # 璁＄畻娑ㄥ仠绫诲瀷鍗犳瘮
                        if "limit_up_count" in board_structure and board_structure["limit_up_count"] > 0:
                            features[f"limit_type_{type_key}_ratio"] = count / board_structure["limit_up_count"]
                
                # 8. 璁＄畻鏉垮潡寮哄害鎸囨爣
                if all(k in board_structure for k in ["avg_pct_chg", "avg_turnover", "avg_amplitude"]):
                    # 缁煎悎寮哄害 = 骞冲潎娑ㄥ箙 * (1 + 骞冲潎鎹㈡墜鐜?100) * (1 + 骞冲潎鎸箙/100)
                    features["sector_strength"] = board_structure["avg_pct_chg"] * (1 + board_structure["avg_turnover"]/100) * (1 + board_structure["avg_amplitude"]/100)
                
                # 9. 璁＄畻鐑害鎸囨暟锛堢畝鍖栫増锛?
                heat_score = 0
                weights = {
                    "limit_up_count": 0.25,          # 娑ㄥ仠鏁伴噺鏉冮噸
                    "limit_up_ratio": 0.15,          # 娑ㄥ仠鍗犳瘮鏉冮噸
                    "max_consecutive_limit_up": 0.25, # 鏈€楂樿繛鏉挎潈閲?
                    "one_word_limit_up_count": 0.15,  # 涓€瀛楁澘鏉冮噸
                    "limit_open_count": 0.10,         # 寮€鐩樻定鍋滄潈閲?
                    "broken_limit_up_count": -0.10    # 鐐告澘鏁伴噺璐熷悜鏉冮噸
                }
                
                for metric, weight in weights.items():
                    if metric in board_structure:
                        if metric == "limit_up_ratio":
                            # 娑ㄥ仠鍗犳瘮鏍囧噯鍖栵紙20%涓烘弧鍒嗭級
                            value = min(100, board_structure.get(metric, 0) * 500)
                        elif metric == "limit_up_count":
                            # 娑ㄥ仠鏁伴噺鏍囧噯鍖栵紙5涓负婊″垎锛?
                            value = min(100, board_structure.get(metric, 0) * 20)
                        elif metric == "max_consecutive_limit_up":
                            # 鏈€楂樿繛鏉挎爣鍑嗗寲锛?0鏉夸负婊″垎锛?
                            value = min(100, board_structure.get(metric, 0) * 10)
                        else:
                            value = board_structure.get(metric, 0)
                            
                        heat_score += value * weight
                
                features["board_heat_index"] = max(0, min(100, heat_score))
                
                # 澧炲姞娑ㄨ穼鍋滄澘楂樼骇鐗瑰緛
                # 杩炴澘鏁伴噺鍒嗗竷
                if "consecutive_limit_counts" in board_structure and isinstance(board_structure["consecutive_limit_counts"], dict):
                    for days, count in board_structure["consecutive_limit_counts"].items():
                        features[f"consecutive_limit_{days}d"] = count

                # 娑ㄥ仠鏉胯川閲忕壒寰?
                if "limit_quality_score" in board_structure:
                    features["limit_quality_score"] = board_structure["limit_quality_score"]
            
            # 濡傛灉board_structure涓篘one锛屽垯涓嶆坊鍔犳定璺屽仠缁撴瀯鐗瑰緛
            # 鍘熸潵鐨勪粠current_day涓彁鍙栫壒寰佺殑閫昏緫宸茶鍒犻櫎
            
            return features
            
        except Exception as e:
            self.logger.error(f"娣诲姞鏉垮潡 {sector_code} 鐨勬定璺屽仠缁撴瀯鐗瑰緛鏃跺嚭閿? {e}")
            return features

    def _calculate_similarity(self, vec1, vec2):
        """璁＄畻涓や釜鍚戦噺鐨勭浉浼煎害
        
        Args:
            vec1: 鍚戦噺1
            vec2: 鍚戦噺2
            
        Returns:
            float: 鐩镐技搴﹀垎鏁?
        """
        # 浣欏鸡鐩镐技搴?
        if np.linalg.norm(vec1) * np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _calculate_feature_importance(self, features_df, target_column, method='correlation'):
        """
        璁＄畻鐗瑰緛閲嶈鎬э紝鐢ㄤ簬鍔犳潈鐩镐技搴﹁绠?
        
        Args:
            features_df: 鐗瑰緛鏁版嵁妗?
            target_column: 鐩爣鍒楀悕锛堝'next_day_return'锛?
            method: 鐗瑰緛閲嶈鎬ц绠楁柟娉曪紝鍙€?correlation'鎴?mutual_info'
            
        Returns:
            dict: 鐗瑰緛閲嶈鎬у瓧鍏革紝閿负鐗瑰緛鍚嶏紝鍊间负閲嶈鎬у垎鏁?
        """
        try:
            # 纭繚鏁版嵁妗嗗寘鍚洰鏍囧垪
            if target_column not in features_df.columns:
                self.logger.error(f"鐩爣鍒?{target_column} 涓嶅瓨鍦ㄤ簬鐗瑰緛鏁版嵁妗嗕腑")
                return {}
                
            # 绉婚櫎闈炴暟鍊煎垪
            numeric_features = features_df.select_dtypes(include=['number']).columns
            numeric_features = [col for col in numeric_features if col != target_column]
            
            # 鍒濆鍖栫壒寰侀噸瑕佹€у瓧鍏?
            feature_importance = {}
            
            if method == 'correlation':
                # 浣跨敤鐩稿叧绯绘暟璁＄畻鐗瑰緛閲嶈鎬?
                for feature in numeric_features:
                    if features_df[feature].nunique() > 1:  # 纭繚鐗瑰緛鏈夊彉鍖?
                        # 璁＄畻鐗瑰緛涓庣洰鏍囩殑缁濆鐩稿叧绯绘暟
                        correlation = abs(features_df[feature].corr(features_df[target_column]))
                        feature_importance[feature] = correlation
            
            elif method == 'mutual_info':
                # 浣跨敤浜掍俊鎭绠楃壒寰侀噸瑕佹€э紙闇€瑕佸畨瑁卻klearn锛?
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    
                    # 濉厖缂哄け鍊?
                    X = features_df[numeric_features].fillna(0)
                    y = features_df[target_column].fillna(0)
                    
                    # 璁＄畻浜掍俊鎭?
                    mi_scores = mutual_info_regression(X, y)
                    
                    # 鍒涘缓鐗瑰緛閲嶈鎬у瓧鍏?
                    for i, feature in enumerate(numeric_features):
                        feature_importance[feature] = mi_scores[i]
                        
                except ImportError:
                    self.logger.warning("鏃犳硶瀵煎叆sklearn锛屽皢浣跨敤鐩稿叧绯绘暟鏂规硶")
                    return self._calculate_feature_importance(features_df, target_column, method='correlation')
            
            else:
                self.logger.warning(f"涓嶆敮鎸佺殑鐗瑰緛閲嶈鎬ц绠楁柟娉? {method}锛屽皢浣跨敤鐩稿叧绯绘暟鏂规硶")
                return self._calculate_feature_importance(features_df, target_column, method='correlation')
            
            # 鏍囧噯鍖栫壒寰侀噸瑕佹€э紝浣挎€诲拰涓?
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= total_importance
            
            # 鎸夐噸瑕佹€ч檷搴忔帓搴?
            feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
            
            self.logger.info(f"鎴愬姛璁＄畻 {len(feature_importance)} 涓壒寰佺殑閲嶈鎬?)
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"璁＄畻鐗瑰緛閲嶈鎬ф椂鍑洪敊: {e}")
            return {}

    def _analyze_feature_contributions(self, pattern_details, top_n=20):
        """
        鍒嗘瀽鐗瑰緛璐＄尞搴︼紝鎵惧嚭瀵圭浉浼煎害鍖归厤鏈€閲嶈鐨勭壒寰?
        
        Args:
            pattern_details: 鍖呭惈鍖归厤妯″紡鍜岀壒寰佽础鐚害鐨勮鎯呭垪琛?
            top_n: 杩斿洖鐨勬渶閲嶈鐗瑰緛鏁伴噺
            
        Returns:
            dict: 鎸夐噸瑕佹€ф帓搴忕殑鐗瑰緛璐＄尞搴﹀瓧鍏?
        """
        try:
            if not pattern_details:
                return {}
                
            # 姹囨€绘墍鏈夌壒寰佺殑璐＄尞搴?
            feature_importance = {}
            
            for pattern in pattern_details:
                if 'feature_contributions' not in pattern:
                    continue
                    
                contributions = pattern['feature_contributions']
                similarity = pattern['similarity']
                
                # 鏍规嵁妯″紡鐩镐技搴﹀姞鏉冪壒寰佽础鐚害
                for feature, contrib in contributions.items():
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    # 绱姞鍔犳潈璐＄尞搴︼紙涔樹互鐩镐技搴︽潈閲嶏級
                    feature_importance[feature] += abs(contrib) * similarity
            
            # 鏍囧噯鍖栫壒寰侀噸瑕佹€?
            if feature_importance:
                total = sum(feature_importance.values())
                if total > 0:
                    for feature in feature_importance:
                        feature_importance[feature] /= total
            
            # 鎸夐噸瑕佹€ч檷搴忔帓搴忓苟闄愬埗鏁伴噺
            sorted_features = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )[:top_n]}
            
            return sorted_features
            
        except Exception as e:
            self.logger.error(f"鍒嗘瀽鐗瑰緛璐＄尞搴︽椂鍑洪敊: {e}")
            return {}

    def _evaluate_selected_stocks(self, selected_stocks, all_stock_features, evaluation_days=2):
        """
        璇勪及閫夊嚭鑲＄エ鐨勫疄闄呰〃鐜帮紝涓庨娴嬬粨鏋滆繘琛屾瘮杈?
        
        Args:
            selected_stocks: 閫夊嚭鐨勮偂绁ㄥ瓧鍏革紝鍖呭惈'selected_stocks_by_date'閿?
            all_stock_features: 鍖呭惈鎵€鏈夎偂绁ㄧ壒寰佺殑DataFrame锛屽寘鎷湭鏉ユ敹鐩婄巼鏁版嵁
            evaluation_days: 璇勪及鐨勫ぉ鏁帮紝榛樿涓?澶╋紙瀵瑰簲future_return_2d锛?
            
        Returns:
            dict: 璇勪及缁撴灉
        """
        self.logger.info(f"寮€濮嬭瘎浼伴€夊嚭鐨勮偂绁ㄥ疄闄呰〃鐜?..")
        
        try:
            # 妫€鏌ユ槸鍚︽湁鏂版牸寮忕殑鎸夋棩鏈熼€夎偂缁撴灉
            has_daily_results = 'selected_stocks_by_date' in selected_stocks and selected_stocks['selected_stocks_by_date']
            
            # 濡傛灉娌℃湁鎸夋棩鏈熷垎缁勭殑缁撴灉锛屽洖閫€鍒版棫鐨勮瘎浼版柟寮?
            if not has_daily_results:
                return self._evaluate_selected_stocks_old(selected_stocks, all_stock_features, evaluation_days)
                
            # 鑾峰彇鎵€鏈夋棩鏈熺殑璇勪及缁撴灉
            evaluation_by_date = {}
            daily_summaries = []
            overall_valid_results = []
            
            for date, stocks in selected_stocks['selected_stocks_by_date'].items():
                # 涓烘瘡涓棩鏈熷垱寤哄吋瀹规棫鏍煎紡鐨勯€夎偂缁撴灉
                daily_selected = {
                    'selected_stocks': stocks,
                    'date': date,
                    'total_stocks': len(stocks)
                }
                
                # 璋冪敤鏃ф柟娉曡瘎浼板崟鏃ラ€夎偂缁撴灉
                daily_eval = self._evaluate_selected_stocks_old(daily_selected, all_stock_features, evaluation_days)
                evaluation_by_date[date] = daily_eval
                
                # 璁板綍姣忔棩璇勪及鎽樿
                if 'evaluation_summary' in daily_eval and isinstance(daily_eval['evaluation_summary'], dict):
                    summary = daily_eval['evaluation_summary']
                    if 'evaluated_stocks' in summary and summary['evaluated_stocks'] > 0:
                        summary['date'] = date  # 娣诲姞鏃ユ湡淇℃伅
                        daily_summaries.append(summary)
                
                # 鏀堕泦鎵€鏈夋棩鏈熺殑鏈夋晥璇勪及缁撴灉鐢ㄤ簬鏁翠綋缁熻
                if 'stock_evaluations' in daily_eval:
                    valid_results = [eval_item for eval_item in daily_eval['stock_evaluations'] 
                                    if 'actual_return' in eval_item and eval_item['actual_return'] is not None]
                    overall_valid_results.extend(valid_results)
            
            # 璁＄畻鏁翠綋璇勪及鎸囨爣
            overall_summary = {}
            if overall_valid_results:
                # 骞冲潎瀹為檯鏀剁泭鐜?
                avg_actual_return = sum(r['actual_return'] for r in overall_valid_results) / len(overall_valid_results)
                
                # 鏂瑰悜姝ｇ‘鐜?
                direction_correct_count = sum(1 for r in overall_valid_results if r.get('direction_correct', False) == True)
                direction_accuracy = direction_correct_count / len(overall_valid_results)
                
                # 璁＄畻骞冲潎棰勬祴鍑嗙‘搴?
                valid_accuracy = [r['prediction_accuracy'] for r in overall_valid_results if r.get('prediction_accuracy') is not None]
                avg_prediction_accuracy = sum(valid_accuracy) / len(valid_accuracy) if valid_accuracy else 0
                
                # 鑳滅巼锛堝疄闄呮敹鐩婁负姝ｇ殑姣斾緥锛?
                win_count = sum(1 for r in overall_valid_results if r['actual_return'] > 0)
                win_rate = win_count / len(overall_valid_results)
                
                overall_summary = {
                    'total_evaluated_dates': len(evaluation_by_date),
                    'total_evaluated_stocks': len(overall_valid_results),
                    'avg_actual_return': avg_actual_return,
                    'direction_accuracy': direction_accuracy,
                    'avg_prediction_accuracy': avg_prediction_accuracy,
                    'win_rate': win_rate,
                    'performance_rating': self._get_performance_rating(avg_actual_return, win_rate, direction_accuracy)
                }
                
                # 璁＄畻鏃ュ钩鍧囪〃鐜?
                if daily_summaries:
                    daily_returns = [s['avg_actual_return'] for s in daily_summaries if 'avg_actual_return' in s]
                    daily_win_rates = [s['win_rate'] for s in daily_summaries if 'win_rate' in s]
                    daily_directions = [s['direction_accuracy'] for s in daily_summaries if 'direction_accuracy' in s]
                    
                    if daily_returns:
                        overall_summary['avg_daily_return'] = sum(daily_returns) / len(daily_returns)
                    if daily_win_rates:
                        overall_summary['avg_daily_win_rate'] = sum(daily_win_rates) / len(daily_win_rates)
                    if daily_directions:
                        overall_summary['avg_daily_direction_accuracy'] = sum(daily_directions) / len(daily_directions)
                
                self.logger.info(f"鏁翠綋璇勪及缁撴灉 - 骞冲潎瀹為檯鏀剁泭鐜? {overall_summary['avg_actual_return']:.2f}%, "
                              f"鏂瑰悜鍑嗙‘鐜? {overall_summary['direction_accuracy']*100:.1f}%, "
                              f"鑳滅巼: {overall_summary['win_rate']*100:.1f}%")
            else:
                overall_summary = {"evaluation_summary": "鏃犳硶璇勪及"}
            
            # 杩斿洖缁煎悎璇勪及缁撴灉
            final_evaluation = {
                'overall_summary': overall_summary,
                'evaluation_by_date': evaluation_by_date,
                'daily_summaries': daily_summaries
            }
            
            return final_evaluation
            
        except Exception as e:
            self.logger.error(f"璇勪及閫夊嚭鑲＄エ鏃跺嚭閿? {e}")
            self.logger.error(traceback.format_exc())
            return {"evaluation_summary": f"璇勪及鍑洪敊: {str(e)}"}
            
    def _evaluate_selected_stocks_old(self, selected_stocks, all_stock_features, evaluation_days=2):
        """
        璇勪及閫夊嚭鑲＄エ鐨勫疄闄呰〃鐜帮紝涓庨娴嬬粨鏋滆繘琛屾瘮杈?(鏃ф柟娉曪紝鐢ㄤ簬鍗曟棩璇勪及)
        
        Args:
            selected_stocks: 閫夊嚭鐨勮偂绁ㄥ垪琛?
            all_stock_features: 鍖呭惈鎵€鏈夎偂绁ㄧ壒寰佺殑DataFrame锛屽寘鎷湭鏉ユ敹鐩婄巼鏁版嵁
            evaluation_days: 璇勪及鐨勫ぉ鏁帮紝榛樿涓?澶╋紙瀵瑰簲future_return_2d锛?
            
        Returns:
            dict: 璇勪及缁撴灉
        """
        stocks_count = len(selected_stocks.get('selected_stocks', []))
        self.logger.info(f"璇勪及鍗曟棩閫夊嚭鐨?{stocks_count} 鍙偂绁ㄧ殑瀹為檯琛ㄧ幇...")
        
        try:
            if not selected_stocks or 'selected_stocks' not in selected_stocks or not selected_stocks['selected_stocks']:
                self.logger.warning("娌℃湁閫夊嚭鐨勮偂绁紝鏃犳硶杩涜璇勪及")
                return {"evaluation_summary": "鏃犻€夊嚭鑲＄エ"}
                
            # 鑾峰彇棰勬祴鏃ユ湡
            prediction_date = selected_stocks.get('date', '')
            if not prediction_date:
                self.logger.warning("鏃犳硶鑾峰彇棰勬祴鏃ユ湡锛屾棤娉曡繘琛岃瘎浼?)
                return {"evaluation_summary": "鏃犻娴嬫棩鏈?}
                
            # 杞崲鏃ユ湡鏍煎紡
            prediction_date = pd.to_datetime(prediction_date)
            
            # 璁＄畻璇勪及鏃ユ湡
            trading_calendar = self._load_trading_calendar()
            if not trading_calendar:
                self.logger.warning("鏃犳硶鍔犺浇浜ゆ槗鏃ュ巻锛屾棤娉曠‘瀹氳瘎浼版棩鏈?)
                return {"evaluation_summary": "鏃犱氦鏄撴棩鍘?}
                
            # 灏嗕氦鏄撴棩鍘嗚浆鎹负鏃ユ湡瀵硅薄鍒楄〃
            trading_dates = [pd.to_datetime(d) for d in trading_calendar]
            trading_dates.sort()
            
            # 鎵惧嚭棰勬祴鏃ユ湡鍦ㄤ氦鏄撴棩鍘嗕腑鐨勪綅缃?
            try:
                date_index = trading_dates.index(prediction_date)
            except ValueError:
                # 鎵惧埌鏈€鎺ヨ繎鐨勪氦鏄撴棩
                closest_date_index = min(range(len(trading_dates)), 
                                        key=lambda i: abs((trading_dates[i] - prediction_date).days))
                date_index = closest_date_index
                self.logger.warning(f"棰勬祴鏃ユ湡 {prediction_date} 涓嶆槸浜ゆ槗鏃ワ紝浣跨敤鏈€杩戠殑浜ゆ槗鏃?{trading_dates[date_index]}")
            
            # 纭繚鏈夎冻澶熺殑鏈潵浜ゆ槗鏃ヨ繘琛岃瘎浼?
            if date_index + evaluation_days >= len(trading_dates):
                self.logger.warning(f"棰勬祴鏃ユ湡 {prediction_date} 鍚庢病鏈夎冻澶熺殑浜ゆ槗鏃ヨ繘琛岃瘎浼?)
                return {"evaluation_summary": "鏃犺冻澶熸湭鏉ヤ氦鏄撴棩"}
                
            # 鑾峰彇璇勪及鏃ユ湡
            evaluation_date = trading_dates[date_index + evaluation_days]
            
            # 鍑嗗璇勪及鏁版嵁
            selected_stock_codes = [stock['stock_code'] for stock in selected_stocks['selected_stocks']]
            
            # 鑾峰彇瀹為檯鏀剁泭鐜囨暟鎹?
            evaluation_data = all_stock_features[
                (all_stock_features['date_str'] == evaluation_date.strftime('%Y%m%d')) & 
                (all_stock_features['stock_code'].isin(selected_stock_codes))
            ].copy()
            
            if evaluation_data.empty:
                self.logger.warning(f"鏃犳硶鑾峰彇璇勪及鏃ユ湡 {evaluation_date} 鐨勮偂绁ㄦ暟鎹?)
                return {"evaluation_summary": "鏃犺瘎浼版暟鎹?}
            
            # 鍚堝苟棰勬祴鍜屽疄闄呮暟鎹?
            evaluation_results = []
            for stock in selected_stocks['selected_stocks']:
                stock_code = stock['stock_code']
                actual_data = evaluation_data[evaluation_data['stock_code'] == stock_code]
                
                if actual_data.empty:
                    # 濡傛灉娌℃湁鎵惧埌瀹為檯鏁版嵁锛岃褰曠己澶?
                    evaluation_results.append({
                        'stock_code': stock_code,
                        'stock_name': stock.get('stock_name', ''),
                        'predicted_return': stock.get('expected_return', 0),
                        'actual_return': None,
                        'prediction_accuracy': None,
                        'rank': stock.get('rank', 0),
                        'status': '鏁版嵁缂哄け'
                    })
                else:
                    # 鑾峰彇瀹為檯鏀剁泭鐜?
                    actual_return = actual_data.iloc[0].get('future_return_2d', None)
                    if actual_return is None:
                        # 灏濊瘯浠庡叾浠栧垪鑾峰彇瀹為檯鏀剁泭鐜?
                        for col in ['return_2d', 'return_2day', 'change_percent_2d']:
                            if col in actual_data.columns:
                                actual_return = actual_data.iloc[0][col]
                                break
                    
                    # 濡傛灉浠嶇劧娌℃湁鎵惧埌瀹為檯鏀剁泭鐜囷紝灏濊瘯璁＄畻鏀剁泭鐜?
                    if actual_return is None and 'close' in actual_data.columns and 'prev_close' in actual_data.columns:
                        try:
                            close = float(actual_data.iloc[0]['close'])
                            prev_close = float(actual_data.iloc[0]['prev_close'])
                            if prev_close > 0:
                                actual_return = (close - prev_close) / prev_close * 100
                            else:
                                actual_return = None
                        except (ValueError, TypeError):
                            actual_return = None
                    
                    # 璁＄畻棰勬祴鍑嗙‘鎬?
                    predicted_return = stock.get('expected_return', 0)
                    if actual_return is not None:
                        # 棰勬祴鏂瑰悜鏄惁姝ｇ‘
                        direction_correct = (predicted_return > 0 and actual_return > 0) or \
                                          (predicted_return < 0 and actual_return < 0)
                        
                        # 璁＄畻棰勬祴璇樊
                        if predicted_return != 0:
                            prediction_error = abs(actual_return - predicted_return) / abs(predicted_return)
                            prediction_accuracy = max(0, 1 - prediction_error)
                        else:
                            prediction_accuracy = 0 if actual_return != 0 else 1
                            
                        # 纭畾鐘舵€?
                        if direction_correct:
                            if prediction_accuracy > 0.7:
                                status = '棰勬祴浼樼'
                            elif prediction_accuracy > 0.4:
                                status = '鏂瑰悜姝ｇ‘'
                            else:
                                status = '鏂瑰悜姝ｇ‘浣嗚宸ぇ'
                        else:
                            status = '鏂瑰悜閿欒'
                    else:
                        direction_correct = None
                        prediction_accuracy = None
                        status = '鏃犳硶璇勪及'
                    
                    # 娣诲姞鍒拌瘎浼扮粨鏋?
                    evaluation_results.append({
                        'stock_code': stock_code,
                        'stock_name': stock.get('stock_name', ''),
                        'predicted_return': predicted_return,
                        'actual_return': actual_return,
                        'direction_correct': direction_correct,
                        'prediction_accuracy': prediction_accuracy,
                        'status': status,
                        'rank': stock.get('rank', 0),
                        'score': stock.get('score', 0)
                    })
            
            # 璁＄畻鎬讳綋璇勪及鎸囨爣
            valid_results = [r for r in evaluation_results if r['actual_return'] is not None]
            if valid_results:
                # 璁＄畻骞冲潎瀹為檯鏀剁泭鐜?
                avg_actual_return = sum(r['actual_return'] for r in valid_results) / len(valid_results)
                
                # 璁＄畻鏂瑰悜姝ｇ‘鐜?
                direction_correct_count = sum(1 for r in valid_results if r['direction_correct'] == True)
                direction_accuracy = direction_correct_count / len(valid_results) if valid_results else 0
                
                # 璁＄畻骞冲潎棰勬祴鍑嗙‘搴?
                avg_prediction_accuracy = sum(r['prediction_accuracy'] for r in valid_results if r['prediction_accuracy'] is not None) / \
                                        len([r for r in valid_results if r['prediction_accuracy'] is not None]) \
                                        if [r for r in valid_results if r['prediction_accuracy'] is not None] else 0
                
                # 璁＄畻鑳滅巼锛堝疄闄呮敹鐩婁负姝ｇ殑姣斾緥锛?
                win_count = sum(1 for r in valid_results if r['actual_return'] > 0)
                win_rate = win_count / len(valid_results) if valid_results else 0
                
                # 璁＄畻璇勪及鎬荤粨
                evaluation_summary = {
                    'prediction_date': prediction_date.strftime('%Y%m%d'),
                    'evaluation_date': evaluation_date.strftime('%Y%m%d') if isinstance(evaluation_date, pd.Timestamp) else str(evaluation_date),
                    'total_stocks': len(selected_stock_codes),
                    'evaluated_stocks': len(valid_results),
                    'avg_actual_return': avg_actual_return,
                    'direction_accuracy': direction_accuracy,
                    'avg_prediction_accuracy': avg_prediction_accuracy,
                    'win_rate': win_rate,
                    'performance_rating': self._get_performance_rating(avg_actual_return, win_rate, direction_accuracy)
                }
            else:
                evaluation_summary = {
                    'prediction_date': prediction_date.strftime('%Y%m%d'),
                    'evaluation_date': evaluation_date.strftime('%Y%m%d') if isinstance(evaluation_date, pd.Timestamp) else str(evaluation_date),
                    'total_stocks': len(selected_stock_codes),
                    'evaluated_stocks': 0,
                    'performance_rating': '鏃犳硶璇勪及'
                }
            
            # 缁勭粐鏈€缁堣瘎浼扮粨鏋?
            final_evaluation = {
                'evaluation_summary': evaluation_summary,
                'stock_evaluations': evaluation_results
            }
            
            self.logger.info(f"璇勪及瀹屾垚锛屽钩鍧囧疄闄呮敹鐩婄巼: {evaluation_summary.get('avg_actual_return', 'N/A')}%, "
                          f"鏂瑰悜鍑嗙‘鐜? {evaluation_summary.get('direction_accuracy', 'N/A') * 100 if 'direction_accuracy' in evaluation_summary else 'N/A'}%, "
                          f"鑳滅巼: {evaluation_summary.get('win_rate', 'N/A') * 100 if 'win_rate' in evaluation_summary else 'N/A'}%")
            
            return final_evaluation
            
        except Exception as e:
            self.logger.error(f"璇勪及閫夊嚭鑲＄エ鏃跺嚭閿? {e}")
            self.logger.error(traceback.format_exc())
            return {"evaluation_summary": f"璇勪及鍑洪敊: {str(e)}"}
    
    def _get_performance_rating(self, avg_return, win_rate, direction_accuracy):
        """
        鏍规嵁骞冲潎鏀剁泭鐜囥€佽儨鐜囧拰鏂瑰悜鍑嗙‘鐜囪瘎瀹氳〃鐜扮瓑绾?
        
        Args:
            avg_return: 骞冲潎鏀剁泭鐜?
            win_rate: 鑳滅巼
            direction_accuracy: 鏂瑰悜鍑嗙‘鐜?
            
        Returns:
            str: 琛ㄧ幇绛夌骇
        """
        # 璇勫垎鏍囧噯
        if avg_return > 3 and win_rate > 0.6 and direction_accuracy > 0.7:
            return '浼樼'
        elif avg_return > 1.5 and win_rate > 0.55 and direction_accuracy > 0.6:
            return '鑹ソ'
        elif avg_return > 0 and win_rate > 0.5 and direction_accuracy > 0.5:
            return '涓€鑸?
        elif avg_return <= 0 and win_rate < 0.5 and direction_accuracy < 0.5:
            return '杈冨樊'
        else:
            return '涓€?

    def _prepare_stock_leadership_features(self, stock_df):
        """
        棰勫鐞嗚偂绁ㄧ殑榫欏ご鐗瑰緛锛屼负璁＄畻鑲＄エ鍦ㄦ澘鍧椾腑鐨勫湴浣嶅仛鍑嗗
        
        Args:
            stock_df: 鑲＄エ鐗瑰緛DataFrame
            
        Returns:
            娣诲姞浜嗛緳澶寸壒寰佺殑DataFrame
        """
        self.logger.info("寮€濮嬮澶勭悊鑲＄エ榫欏ご鐗瑰緛...")
        
        if stock_df.empty:
            return stock_df
            
        result_df = stock_df.copy()
        
        # 1. 璇嗗埆杩炴澘澶╂暟
        if 'is_limit_up' in result_df.columns:
            # 鍒濆鍖栬繛缁定鍋滃ぉ鏁板垪
            result_df['consecutive_limit_days'] = 0
            
            # 鎸夎偂绁ㄤ唬鐮佸垎缁勫鐞?
            for stock_code, group in result_df.groupby('stock_code'):
                # 纭繚鎸夋棩鏈熸帓搴?
                group = group.sort_values('date_str')
                
                # 璁＄畻杩炵画娑ㄥ仠澶╂暟
                consecutive_days = 0
                for idx, row in group.iterrows():
                    if row['is_limit_up']:
                        consecutive_days += 1
                    else:
                        consecutive_days = 0
                    
                    # 鏇存柊杩炵画娑ㄥ仠澶╂暟
                    result_df.loc[idx, 'consecutive_limit_days'] = consecutive_days
        
        # 2. 璇嗗埆涓€瀛楁澘
        if all(col in result_df.columns for col in ['open', 'high', 'low', 'close', 'is_limit_up']):
            # 鍒ゆ柇寮€鐩樸€佹渶楂樸€佹渶浣庛€佹敹鐩樹环鏍兼槸鍚︾浉鍚岋紝涓旀槸娑ㄥ仠
            result_df['is_one_word_limit'] = (
                (result_df['open'] == result_df['high']) & 
                (result_df['high'] == result_df['low']) & 
                (result_df['low'] == result_df['close']) & 
                result_df['is_limit_up']
            )
        else:
            result_df['is_one_word_limit'] = False
            
        # 3. 璇嗗埆楂樺脊鎬ц偂绁?
        if 'amplitude' in result_df.columns:
            # 鎸箙澶т簬5%涓旈潪娑ㄨ穼鍋滅殑鑲＄エ瑙嗕负楂樺脊鎬?
            result_df['is_high_elasticity'] = (
                (result_df['amplitude'] >= 5.0) & 
                (~result_df['is_limit_up']) & 
                (~result_df.get('is_limit_down', pd.Series(False, index=result_df.index)))
            )
        else:
            result_df['is_high_elasticity'] = False
            
        # 4. 鍒ゆ柇甯傚€肩被鍨?
        if 'total_mv' in result_df.columns:
            # 鏍规嵁鎬诲競鍊煎垝鍒嗗ぇ涓皬甯傚€?
            result_df['market_cap_type'] = 'small'  # 榛樿灏忓競鍊?
            result_df.loc[result_df['total_mv'] > 50, 'market_cap_type'] = 'medium'  # 50-200浜夸负涓競鍊?
            result_df.loc[result_df['total_mv'] > 200, 'market_cap_type'] = 'large'  # >200浜夸负澶у競鍊?
        else:
            result_df['market_cap_type'] = 'small'
            
        # 5. 鍒ゆ柇鏄惁寮€鐩樻定鍋?
        if all(col in result_df.columns for col in ['open', 'pre_close', 'is_limit_up']):
            # ST鑲＄エ娑ㄥ仠骞呭害鏄?%锛岄潪ST鏄?0%
            # 杩欓噷绠€鍖栧鐞嗭紝缁熶竴鎸夌収9.5%鐨勯槇鍊煎垽鏂槸鍚﹀紑鐩樻定鍋?
            result_df['is_open_limit_up'] = (
                (result_df['open'] / result_df['pre_close'] - 1 > 0.095) & 
                result_df['is_limit_up']
            )
        else:
            result_df['is_open_limit_up'] = False
            
        # 6. 鍒ゆ柇鏄惁绐佺牬鍓嶆湡楂樼偣
        if 'close' in result_df.columns:
            # 鎸夎偂绁ㄤ唬鐮佸垎缁勫鐞?
            for stock_code, group in result_df.groupby('stock_code'):
                # logger.info(f"鍒楋細{group.columns}")
                # 纭繚鎸夋棩鏈熸帓搴?
                group = group.sort_values('date_str')
                
                # 璁＄畻60鏃ユ渶楂樹环
                group['max_60d'] = group['close'].rolling(window=60, min_periods=1).max().shift(1)
                
                # 鍒ゆ柇鏄惁绐佺牬鍓嶆湡楂樼偣
                group['break_previous_high'] = group['close'] > group['max_60d']
                
                # 鏇存柊鍥炲師濮婦ataFrame
                result_df.loc[group.index, 'max_60d'] = group['max_60d']
                result_df.loc[group.index, 'break_previous_high'] = group['break_previous_high']
        else:
            result_df['break_previous_high'] = False
            
        return result_df
        
    def _process_single_stock_date(self, stock_code, date_str, result_df_idx, stock_sector_mapping, sector_features_df_idx):
        """澶勭悊鍗曚釜鑲＄エ鏃ユ湡缁勫悎鐨勭壒寰佺浉鍏虫€?
        
        Args:
            stock_code: 鑲＄エ浠ｇ爜
            date_str: 鏃ユ湡瀛楃涓?
            result_df_idx: 鑲＄エ鐗瑰緛DataFrame绱㈠紩
            stock_sector_mapping: 鑲＄エ涓庢澘鍧楃殑鏄犲皠瀛楀吀
            sector_features_df_idx: 鏉垮潡鐗瑰緛DataFrame绱㈠紩
            
        Returns:
            dict: 澶勭悊缁撴灉锛屽鏋滄棤鏁堝垯杩斿洖None
        """
        try:
            # 浣跨敤绱㈠紩蹇€熻幏鍙栨暟鎹?
            try:
                stock_row = result_df_idx.loc[(stock_code, date_str)]
            except KeyError:
                return None
            
            # 鏌ユ壘鑲＄エ鎵€灞炴澘鍧?
            sectors = stock_sector_mapping.get(stock_code, [])
            if not sectors:
                return None
            
            # 浣跨敤鍚戦噺鍖栨搷浣滆幏鍙栫浉鍏虫澘鍧楁暟鎹?
            sector_rows = []
            sector_codes = []
            for sector_code in sectors:
                try:
                    sector_row = sector_features_df_idx.loc[(sector_code, date_str)]
                    sector_rows.append(sector_row)
                    sector_codes.append(sector_code)
                except KeyError:
                    continue
            
            if not sector_rows:
                return None
            
            # 浼樺寲璁＄畻锛氶鍏堣幏鍙栬偂绁ㄧ壒寰侊紝閬垮厤閲嶅璁块棶
            stock_price = stock_row.get('close', 0)
            stock_change = stock_row.get('pct_chg', 0)
            stock_volume = stock_row.get('vol', 0)
            stock_turnover = stock_row.get('turnover_rate', 0)
            stock_amplitude = stock_row.get('amplitude', 0)
            
            # 纭繚鍊间负鏍囬噺锛屼慨澶?Series truth value 闂
            if isinstance(stock_change, pd.Series):
                stock_change = stock_change.iloc[0] if not stock_change.empty else 0
            if isinstance(stock_volume, pd.Series):
                stock_volume = stock_volume.iloc[0] if not stock_volume.empty else 0
            if isinstance(stock_turnover, pd.Series):
                stock_turnover = stock_turnover.iloc[0] if not stock_turnover.empty else 0
            if isinstance(stock_amplitude, pd.Series):
                stock_amplitude = stock_amplitude.iloc[0] if not stock_amplitude.empty else 0
            
            # 璁＄畻涓庡悇鏉垮潡鐨勭浉鍏虫€ц瘎鍒嗭紙鍚戦噺鍖栧鐞嗭級
            sector_scores = []
            
            for i, sector_row in enumerate(sector_rows):
                sector_code = sector_codes[i]
                
                # 1. 娑ㄨ穼骞呯浉鍏虫€?
                sector_change = sector_row.get('pct_chg', 0)
                if isinstance(sector_change, pd.Series):
                    sector_change = sector_change.iloc[0] if not sector_change.empty else 0
                    
                return_correlation = 0.0
                # 淇: 娣诲姞绫诲瀷瀹夊叏妫€鏌ュ拰isfinite妫€鏌?
                try:
                    if (isinstance(stock_change, (int, float, np.number)) and 
                        isinstance(sector_change, (int, float, np.number)) and
                        np.isfinite(stock_change) and np.isfinite(sector_change)):
                        return_diff = abs(stock_change - sector_change)
                        return_correlation = 1.0 / (1.0 + return_diff)
                except Exception as e:
                    # 瀹夊叏澶辫触锛屼繚鎸侀粯璁ゅ€?
                    return_correlation = 0.0
                
                # 2. 鎴愪氦閲忕浉鍏虫€?
                volume_correlation = 0.0
                sector_volume = sector_row.get('vol', 0)
                if isinstance(sector_volume, pd.Series):
                    sector_volume = sector_volume.iloc[0] if not sector_volume.empty else 0
                    
                if stock_volume > 0 and sector_volume > 0:
                    vol_ratio = min(stock_volume, sector_volume) / max(stock_volume, sector_volume)
                    volume_correlation = vol_ratio
                
                # 3. 鎹㈡墜鐜囩浉鍏虫€?
                turnover_correlation = 0.0
                sector_turnover = sector_row.get('turnover_rate', 0)
                if isinstance(sector_turnover, pd.Series):
                    sector_turnover = sector_turnover.iloc[0] if not sector_turnover.empty else 0
                    
                if stock_turnover > 0 and sector_turnover > 0:
                    turnover_ratio = min(stock_turnover, sector_turnover) / max(stock_turnover, sector_turnover)
                    turnover_correlation = turnover_ratio
                
                # 4. 娉㈠姩鐜囩浉鍏虫€?
                volatility_correlation = 0.0
                sector_amplitude = sector_row.get('amplitude', 0)
                if isinstance(sector_amplitude, pd.Series):
                    sector_amplitude = sector_amplitude.iloc[0] if not sector_amplitude.empty else 0
                    
                if stock_amplitude > 0 and sector_amplitude > 0:
                    amp_ratio = min(stock_amplitude, sector_amplitude) / max(stock_amplitude, sector_amplitude)
                    volatility_correlation = amp_ratio
                
                # 5. 鎶€鏈寚鏍囩浉鍏虫€?
                tech_correlation = 0.0
                tech_indicators = ['rsi14', 'ma5_slope', 'momentum_10d']
                matching_indicators = []
                
                # 淇锛氭鏌ュ瓨鍦ㄧ殑鎶€鏈寚鏍?
                for ind in tech_indicators:
                    has_stock_ind = ind in stock_row.index
                    has_sector_ind = ind in sector_row.index
                    if has_stock_ind and has_sector_ind:
                        matching_indicators.append(ind)
                
                if matching_indicators:
                    tech_diffs = []
                    for ind in matching_indicators:
                        stock_val = stock_row[ind]
                        sector_val = sector_row[ind]
                        
                        # 纭繚鍊间负鏍囬噺
                        if isinstance(stock_val, pd.Series):
                            stock_val = stock_val.iloc[0] if not stock_val.empty else 0
                        if isinstance(sector_val, pd.Series):
                            sector_val = sector_val.iloc[0] if not sector_val.empty else 0
                            
                        # 纭繚鏄暟鍊硷紝閬垮厤姣旇緝閿欒
                        if pd.notna(stock_val) and pd.notna(sector_val):
                            tech_diffs.append(abs(stock_val - sector_val))
                            
                    avg_diff = sum(tech_diffs) / len(tech_diffs) if tech_diffs else 1.0
                    tech_correlation = 1.0 / (1.0 + avg_diff)
                
                # 6. 涓偂鍦ㄦ澘鍧椾腑鐨勫湴浣?
                leadership_score = 0.0
                is_leader = False
                
                # 浣跨敤宸茶绠楃殑榫欏ご鐗瑰緛
                consecutive_days = stock_row.get('consecutive_limit_days', 0)
                if isinstance(consecutive_days, pd.Series):
                    consecutive_days = consecutive_days.iloc[0] if not consecutive_days.empty else 0
                
                if consecutive_days >= 2:
                    leadership_score += min(1.0, consecutive_days / 3.0)
                    is_leader = True
                
                # 纭繚甯冨皵鍊兼槸鏍囬噺
                is_one_word_limit = stock_row.get('is_one_word_limit', False)
                if isinstance(is_one_word_limit, pd.Series):
                    is_one_word_limit = is_one_word_limit.iloc[0] if not is_one_word_limit.empty else False
                    
                if is_one_word_limit:
                    leadership_score += 0.5
                    is_leader = True
                
                is_high_elasticity = stock_row.get('is_high_elasticity', False)
                if isinstance(is_high_elasticity, pd.Series):
                    is_high_elasticity = is_high_elasticity.iloc[0] if not is_high_elasticity.empty else False
                    
                if is_high_elasticity:
                    leadership_score += 0.3
                
                market_cap_type = stock_row.get('market_cap_type', '')
                if isinstance(market_cap_type, pd.Series):
                    market_cap_type = market_cap_type.iloc[0] if not market_cap_type.empty else ''
                
                is_limit_up = stock_row.get('is_limit_up', False)
                if isinstance(is_limit_up, pd.Series):
                    is_limit_up = is_limit_up.iloc[0] if not is_limit_up.empty else False
                    
                if market_cap_type == 'large' and (is_limit_up or consecutive_days > 0):
                    leadership_score += 0.4
                    is_leader = True
                
                leadership_score = min(1.0, leadership_score)
                
                # 7. 娑ㄨ穼鍋滅粨鏋勭浉浼兼€?
                limit_structure_score = 0.0
                
                # 淇锛氬畨鍏ㄥ湴妫€鏌ュ瓧娈垫槸鍚﹀瓨鍦?
                limit_up_count_exists = 'limit_up_count' in sector_row.index
                limit_down_count_exists = 'limit_down_count' in sector_row.index
                
                has_limit_up_count = False
                has_limit_down_count = False
                
                if limit_up_count_exists:
                    limit_up_count = sector_row['limit_up_count']
                    if isinstance(limit_up_count, pd.Series):
                        limit_up_count = limit_up_count.iloc[0] if not limit_up_count.empty else 0
                    has_limit_up_count = limit_up_count > 0
                    
                if limit_down_count_exists:
                    limit_down_count = sector_row['limit_down_count']
                    if isinstance(limit_down_count, pd.Series):
                        limit_down_count = limit_down_count.iloc[0] if not limit_down_count.empty else 0
                    has_limit_down_count = limit_down_count > 0
                
                if has_limit_up_count or has_limit_down_count:
                    is_stock_limit_up = stock_row.get('is_limit_up', False)
                    if isinstance(is_stock_limit_up, pd.Series):
                        is_stock_limit_up = is_stock_limit_up.iloc[0] if not is_stock_limit_up.empty else False
                        
                    is_stock_limit_down = stock_row.get('is_limit_down', False)
                    if isinstance(is_stock_limit_down, pd.Series):
                        is_stock_limit_down = is_stock_limit_down.iloc[0] if not is_stock_limit_down.empty else False
                    
                    if is_stock_limit_up and has_limit_up_count:
                        limit_structure_score += 0.5
                    if is_stock_limit_down and has_limit_down_count:
                        limit_structure_score += 0.5
                
                # 璁＄畻鍔犳潈鎬诲垎 - 浣跨敤鏇撮珮鏁堢殑鏉冮噸璁＄畻鏂瑰紡
                # 棰勫畾涔夋潈閲嶏紝閬垮厤閲嶅鍒涘缓瀛楀吀
                return_weight = 0.3
                volume_weight = 0.15
                turnover_weight = 0.15
                volatility_weight = 0.1
                tech_weight = 0.1
                leadership_weight = 0.1
                limit_weight = 0.1
                
                # 鐩存帴璁＄畻鍔犳潈鍜?
                total_score = (
                    return_correlation * return_weight +
                    volume_correlation * volume_weight +
                    turnover_correlation * turnover_weight +
                    volatility_correlation * volatility_weight +
                    tech_correlation * tech_weight +
                    leadership_score * leadership_weight +
                    limit_structure_score * limit_weight
                )
                
                sector_scores.append({
                    'sector_code': sector_code,
                    'total_score': total_score,
                    'return_correlation': return_correlation,
                    'volume_correlation': volume_correlation,
                    'turnover_correlation': turnover_correlation, 
                    'volatility_correlation': volatility_correlation,
                    'tech_correlation': tech_correlation,
                    'leadership_score': leadership_score,
                    'limit_structure_score': limit_structure_score
                })
            
            # 濡傛灉鑷冲皯鏈変竴涓湁鏁堢殑鏉垮潡鐩稿叧鎬э紝娣诲姞鍒扮粨鏋滀腑
            if sector_scores:
                # 鎵惧嚭鏈€楂樺垎鍜屾楂樺垎鐨勬澘鍧?
                sorted_scores = sorted(sector_scores, key=lambda x: x['total_score'], reverse=True)
                best_sector = sorted_scores[0]
                second_best = sorted_scores[1] if len(sorted_scores) > 1 else {'sector_code': '', 'total_score': 0.0}
                
                # 杩斿洖缁撴灉
                return {
                    'stock_code': stock_code,
                    'date': date_str,
                    'best_sector': best_sector['sector_code'],
                    'best_score': best_sector['total_score'],
                    'best_return_corr': best_sector['return_correlation'],
                    'best_volume_corr': best_sector['volume_correlation'],
                    'best_turnover_corr': best_sector['turnover_correlation'],
                    'best_volatility_corr': best_sector['volatility_correlation'],
                    'best_tech_corr': best_sector['tech_correlation'],
                    'best_leadership': best_sector['leadership_score'],
                    'best_limit_struct': best_sector['limit_structure_score'],
                    'second_sector': second_best['sector_code'],
                    'second_score': second_best['total_score']
                }
            
            return None
        except Exception as e:
            self.logger.error(f"澶勭悊鑲＄エ {stock_code} 鍦?{date_str} 鐨勭壒寰佹椂鍑洪敊: {str(e)}")
            # 娣诲姞璇︾粏鐨勯敊璇窡韪互渚挎洿濂藉湴璋冭瘯
            import traceback
            self.logger.error(f"閿欒璇︽儏: {traceback.format_exc()}")
            return None


def main():
    """涓诲嚱鏁帮紝鐢ㄤ簬鍛戒护琛岃皟鐢?""
    parser = argparse.ArgumentParser(description="鏉垮潡杞姩妯″瀷璁粌涓庨娴?)
    parser.add_argument('--mode', type=str, choices=['train', 'select_stocks'], 
                      help='鎿嶄綔妯″紡锛歵rain (璁粌妯″瀷), select_stocks (閫夎偂)')
    parser.add_argument('--start_date', type=str, help='寮€濮嬫棩鏈燂紝鏍煎紡涓篩YYYMMDD')
    parser.add_argument('--end_date', type=str, help='缁撴潫鏃ユ湡锛屾牸寮忎负YYYYMMDD')
    parser.add_argument('--date', type=str, help='棰勬祴鏃ユ湡锛屾牸寮忎负YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=50, help='閫夋嫨鍓峃涓澘鍧楁垨鑲＄エ')
    parser.add_argument('--max_workers', type=int, help='骞惰璁粌鐨勮繘绋嬫暟')
    parser.add_argument('--sector_list', type=str, help='鏉垮潡鍒楄〃鏂囦欢璺緞锛岀敤浜庢澘鍧楄疆鍔ㄦā鍨嬭缁?)
    parser.add_argument('--model_dir', type=str, help='妯″瀷淇濆瓨鐩綍')
    parser.add_argument('--model_name', type=str, default='sector_rotation_model', help='妯″瀷鍚嶇О')
    parser.add_argument('--save_results', type=str, help='淇濆瓨缁撴灉鐨勬枃浠惰矾寰?)
    parser.add_argument('--stock_list', type=str, help='鑲＄エ鍒楄〃鏂囦欢璺緞锛岀敤浜庨€夎偂')
    parser.add_argument('--use_cache', action='store_true', help='鏄惁浣跨敤鐗瑰緛缂撳瓨锛岄粯璁や负True')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false', help='涓嶄娇鐢ㄧ壒寰佺紦瀛?)
    parser.set_defaults(use_cache=True)
    
    args = parser.parse_args()
    print(f"寮€濮?)
   
    logger.setLevel(logging.DEBUG)
    # 鍒涘缓SectorModelTrainer瀹炰緥
    trainer = SectorModelTrainer(model_dir=args.model_dir)
    
    # 鍔犺浇鏉垮潡鍒楄〃
    sector_list = []
    try:
        # 浼樺厛浣跨敤鍙傛暟鎸囧畾鐨勬澘鍧楀垪琛ㄦ枃浠?
        if args.sector_list and os.path.exists(args.sector_list):
            with open(args.sector_list, 'r', encoding='utf-8') as f:
                sector_list = [line.strip() for line in f if line.strip()]
            logger.info(f"浠庢枃浠?{args.sector_list} 鍔犺浇浜?{len(sector_list)} 涓澘鍧?)
        else:
            # 灏濊瘯浠巗tock_sectors_mapping妯″潡鑾峰彇
            try:
                # 灏嗗綋鍓嶅伐浣滅洰褰曟坊鍔犲埌sys.path
                sys.path.append(os.getcwd())
                from stock_sectors_mapping import get_tng_sectors
                sector_list = get_tng_sectors()
                logger.info(f"浠巊et_tng_sectors鍔犺浇浜?{len(sector_list)} 涓澘鍧?)
            except ImportError:
                # 濡傛灉瀵煎叆澶辫触锛屽皾璇曚粠to鐩綍瀵煎叆
                try:
                    sys.path.append(str(get_project_root()))
                    from to.stock_sectors_mapping import get_tng_sectors
                    sector_list = get_tng_sectors()
                    logger.info(f"浠巘o.stock_sectors_mapping鍔犺浇浜?{len(sector_list)} 涓澘鍧?)
                except ImportError:
                    logger.error("鏃犳硶瀵煎叆get_tng_sectors鍑芥暟锛岃鎻愪緵鏉垮潡鍒楄〃鏂囦欢")
    except Exception as e:
        logger.error(f"鑾峰彇鏉垮潡鍒楄〃鏃跺嚭閿? {e}")
        logger.error(traceback.format_exc())
    
    # 鍔犺浇鑲＄エ鍒楄〃锛堝鏋滈渶瑕侊級
    stock_list = None
    if args.stock_list and os.path.exists(args.stock_list):
        try:
            with open(args.stock_list, 'r', encoding='utf-8') as f:
                stock_list = [line.strip() for line in f if line.strip()]
            logger.info(f"浠庢枃浠?{args.stock_list} 鍔犺浇浜?{len(stock_list)} 鍙偂绁?)
        except Exception as e:
            logger.error(f"鍔犺浇鑲＄エ鍒楄〃鏃跺嚭閿? {e}")
    
    # 杩愯妯″紡
    if args.mode == 'train':
        # 璁粌鏉垮潡杞姩妯″瀷锛堝鏋滃凡瀛樺湪妯″瀷锛屼細鍏堝皾璇曞姞杞斤級
        if not sector_list:
            print("閿欒: 璁粌鏉垮潡杞姩妯″瀷闇€瑕佹彁渚涙澘鍧楀垪琛?)
            return
                    
        # 璋冪敤璁粌鍑芥暟 - 鍐呴儴宸插寘鍚簡2025骞?鏈堢敓鍛藉懆鏈熼娴嬪拰缁撴灉淇濆瓨
        model_result = trainer.train_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
                model_name=args.model_name)
            
        if model_result:
            print(f"鏉垮潡杞姩妯″瀷璁粌鎴愬姛锛?025骞?鏈堢殑鐢熷懡鍛ㄦ湡闃舵棰勬祴缁撴灉宸蹭繚瀛樺湪results鐩綍涓?)
        else:
            print("鏉垮潡杞姩妯″瀷璁粌澶辫触")
    
    elif args.mode == 'select_stocks':
        # 浣跨敤鍙樺寲鐐规柟娉曡繘琛岄€夎偂
        top_n = args.top_n if args.top_n else 50
        model_name = args.model_name if args.model_name else "stock_selector"
        
        # 璁剧疆鏃ユ湡鑼冨洿
        start_date = args.start_date
        end_date = args.end_date if args.end_date else datetime.now().strftime("%Y%m%d")
        
        print(f"寮€濮嬭繍琛屽熀浜庡彉鍖栫偣妫€娴嬬殑鏅鸿兘閫夎偂妗嗘灦...")
        print(f"- 鏃堕棿鑼冨洿: {start_date} 鑷?{end_date}")
        print(f"- 閫夊嚭鏁伴噺: 鍓?{top_n} 鍙偂绁?)
        print(f"- 妯″瀷鍚嶇О: {model_name}")
        print(f"- 浣跨敤缂撳瓨: {'鏄? if args.use_cache else '鍚?}")
        print("-" * 80)
        
        # 璋冪敤閫夎偂鍑芥暟
        selected_stocks = trainer.select_stocks_by_change_points(
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
            top_n=top_n,
            use_cache=args.use_cache
        )
        
        if selected_stocks:
            # 鎵撳嵃閫夎偂缁撴灉
            print(f"\n閫夎偂缁撴灉 (鏃ユ湡: {selected_stocks['date']}, 鍏遍€夊嚭 {len(selected_stocks['selected_stocks'])} 鍙偂绁?:")
            print("-" * 80)
            print(f"{'鎺掑悕':<6}{'鑲＄エ浠ｇ爜':<12}{'鑲＄エ鍚嶇О':<15}{'棰勬湡鏀剁泭鐜?:>12}{'姒傜巼':>10}{'缃俊搴?:>10}{'寰楀垎':>10}")
            print("-" * 80)
            
            for i, stock in enumerate(selected_stocks['selected_stocks']):
                if i < 20:  # 鍙樉绀哄墠20鍙?
                    print(f"{i+1:<6}{stock['stock_code']:<12}{stock['stock_name']:<15}{stock['expected_return']*100:>12.2f}%{stock['probability']*100:>9.1f}%{stock['confidence']*100:>9.1f}%{stock['score']*100:>9.2f}")
            
            if len(selected_stocks['selected_stocks']) > 20:
                print(f"... 杩樻湁 {len(selected_stocks['selected_stocks'])-20} 鍙偂绁ㄦ湭鏄剧ず")
                
            print(f"\n閫夎偂缁撴灉宸蹭繚瀛樺埌 {trainer.model_dir / f'{model_name}_results.json'}")
        else:
            print("閫夎偂澶辫触锛岃妫€鏌ユ棩蹇楄幏鍙栬缁嗕俊鎭?)
    else:
        print("璇锋寚瀹氭湁鏁堢殑杩愯妯″紡: --mode train 鎴?--mode select_stocks")
        print("\n绀轰緥鐢ㄦ硶:")
        print("璁粌鏉垮潡杞姩妯″瀷:")
        print("  python train_sector_models.py --mode train --start_date 20230101 --end_date 20231231 --model_name sector_rotation_model")
        print("\n鏅鸿兘閫夎偂:")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --model_name stock_selector")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --no_cache  # 涓嶄娇鐢ㄧ紦瀛?)


if __name__ == "__main__":
    main() 
