*** Begin Patch
*** Update File: to/integrated_stock_selection/data_loader.py
@@
-    def load_sector_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
-        """���ذ����������"""
-        try:
-            self.logger.info(f"���ذ����������: {start_date} - {end_date}")
-
-            # ʹ�û����������İ���������ط���
-            sector_data = self.base_loader.load_sector_features(start_date, end_date)
-
-            if sector_data is None:
-                self.logger.error("�����������ʧ��")
-                return None
-
-            self.logger.info(f"�ɹ����ذ������: {len(sector_data)}����¼, {sector_data['sector_code'].nunique()}�����")
-
-            return sector_data
-
-        except Exception as e:
-            self.logger.error(f"���ذ������ʧ��: {e}")
-            return None
+    def load_sector_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
+        """���ذ����������"""
+        try:
+            self.logger.info(f"���ذ����������: {start_date} - {end_date}")
+
+            sector_data = None
+            try:
+                sector_data = self.base_loader.load_sector_features(start_date, end_date)
+            except Exception as inner_exc:
+                self.logger.warning(f"רҵ板块加载失败，将尝试简化逻辑: {inner_exc}")
+
+            sector_data = self._post_process_sector_data(sector_data, start_date, end_date)
+
+            if sector_data is None or sector_data.empty:
+                self.logger.warning("板块数据为空或缺少关键字段，启用简化加载逻辑")
+                sector_data = self._load_sector_features_basic(start_date, end_date)
+
+            if sector_data is None or sector_data.empty:
+                self.logger.error("板块数据加载失败")
+                return None
+
+            sector_count = sector_data['sector_code'].nunique() if 'sector_code' in sector_data.columns else '未知'
+            self.logger.info(f"成功加载板块数据: {len(sector_data)} 条记录, {sector_count} 个板块")
+
+            return sector_data
+
+        except Exception as e:
+            self.logger.error(f"板块数据加载失败: {e}")
+            return None
+
+    def _post_process_sector_data(
+        self,
+        data: Optional[pd.DataFrame],
+        start_date: str,
+        end_date: str,
+        default_sector_code: Optional[str] = None,
+    ) -> Optional[pd.DataFrame]:
+        """保证板块数据具备因子计算所需字段"""
+        if data is None:
+            return None
+
+        if data.empty:
+            return data
+
+        try:
+            df = data.copy()
+            start_dt = pd.to_datetime(start_date)
+            end_dt = pd.to_datetime(end_date)
+
+            if 'date' not in df.columns:
+                if 'date_str' in df.columns:
+                    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
+                else:
+                    self.logger.warning("板块数据缺少 date 字段，无法使用")
+                    return None
+            else:
+                df['date'] = pd.to_datetime(df['date'], errors='coerce')
+
+            df = df.dropna(subset=['date'])
+            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
+            if df.empty:
+                return df
+
+            if 'sector_code' not in df.columns:
+                if 'sector' in df.columns:
+                    df['sector_code'] = df['sector']
+                elif 'sector_name' in df.columns:
+                    df['sector_code'] = df['sector_name']
+                elif default_sector_code is not None:
+                    df['sector_code'] = default_sector_code
+                else:
+                    self.logger.warning("板块数据缺少 sector_code，且无法推断")
+                    return None
+
+            df['sector_code'] = df['sector_code'].astype(str)
+            if 'sector_name' not in df.columns and 'sector' in df.columns:
+                df['sector_name'] = df['sector']
+
+            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
+            df = df.sort_values(['sector_code', 'date']).reset_index(drop=True)
+
+            if 'pct_chg' in df.columns:
+                df['return_1d'] = df['pct_chg'].astype(float)
+            else:
+                df['return_1d'] = df.groupby('sector_code')['close'].pct_change() * 100
+
+            df['return'] = df['return_1d']
+            df['daily_return'] = df['return_1d']
+
+            if 'close' in df.columns:
+                group_close = df.groupby('sector_code')['close']
+                for window in (3, 5, 10, 20):
+                    col = f'return_{window}d'
+                    if col not in df.columns:
+                        df[col] = group_close.transform(lambda s, w=window: (s / s.shift(w) - 1.0) * 100)
+            else:
+                self.logger.warning("板块数据缺少 close 列，无法计算多日收益")
+
+            # 若仍缺少 return_1d 相关列，回退为 0
+            for col in ['return', 'daily_return', 'return_1d']:
+                if col not in df.columns:
+                    df[col] = 0.0
+
+            return df
+
+        except Exception as exc:
+            self.logger.warning(f"板块数据整理失败: {exc}")
+            return None
+
+    def _load_sector_features_basic(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
+        """简化的板块数据加载逻辑，不强依赖额外衍生文件"""
+        try:
+            sector_root = system_config.sector_features_dir
+            if not sector_root.exists():
+                self.logger.warning(f"板块特征目录不存在: {sector_root}")
+                return None
+
+            frames = []
+            for sector_dir in sector_root.iterdir():
+                if not sector_dir.is_dir():
+                    continue
+                sector_code = sector_dir.name
+
+                candidates = [
+                    sector_dir / f"{sector_code}_features.parquet",
+                    sector_dir / f"{sector_code}.parquet",
+                ]
+                feature_file = next((p for p in candidates if p.exists()), None)
+                if feature_file is None:
+                    continue
+
+                try:
+                    df = pd.read_parquet(feature_file)
+                except Exception as read_exc:
+                    self.logger.debug(f"读取板块文件失败 {feature_file}: {read_exc}")
+                    continue
+
+                processed = self._post_process_sector_data(df, start_date, end_date, default_sector_code=sector_code)
+                if processed is not None and not processed.empty:
+                    frames.append(processed)
+
+            if not frames:
+                self.logger.warning(f"简化加载仍未得到板块数据: {start_date} - {end_date}")
+                return None
+
+            sector_data = pd.concat(frames, ignore_index=True)
+            sector_data = sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)
+            self.logger.info(f"简化板块加载成功: {len(sector_data)} 条记录, {sector_data['sector_code'].nunique()} 个板块")
+            return sector_data
+
+        except Exception as exc:
+            self.logger.error(f"简化加载板块数据失败: {exc}")
+            return None
*** End Patch
