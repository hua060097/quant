from pathlib import Path
from textwrap import dedent
path = Path(r"c:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/data_loader.py")
text = path.read_text(encoding="utf-8")
marker = dedent('''
    def load_stock_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """���ظ�����������"""
        try:
            self.logger.info(f"���ظ�����������: {start_date} - {end_date}")
            
            # ��黺��
            cache_key = f"stock_features_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                self.logger.info(f"�ӻ������: {cache_file}")
                return pd.read_parquet(cache_file)
            
            # ʹ�ñ���ʵ����ȷ����ȷ������·��
            # ��ȡ������
            trading_dates = self.get_trading_dates(start_date, end_date)
            if not trading_dates:
                self.logger.error("û���ҵ�������")
                return None

            all_data = []

            # ���ռ��������ļ�
            # ʹ������ڵ�ǰ�ļ���·����ȷ��·����ȷ
            current_file_dir = Path(__file__).parent  # integrated_stock_selectionĿ¼
            features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

            for date_str in trading_dates:
                # �����ն��ļ�·����local_data/features/daily/YYYYMMDD.parquet
                file_path = features_dir / f"{date_str}.parquet"

                if not file_path.exists():
                    self.logger.warning(f"�ն��ļ�������: {file_path}")
                    continue

                try:
                    # �����ն�����
                    daily_data = pd.read_parquet(file_path)

                    # ȷ����������
                    if 'date' not in daily_data.columns:
                        # ���û��date�У���������
                        daily_data['date'] = pd.to_datetime(date_str)

                    all_data.append(daily_data)
                    self.logger.info(f"����{date_str}����: {len(daily_data)}����¼")

                except Exception as e:
                    self.logger.error(f"�����ն��ļ�ʧ�� {file_path}: {e}")
                    continue
            
            if not all_data:
                self.logger.error("û�гɹ������κ�����")
                self.logger.error(f"����������: {len(trading_dates)}")
                self.logger.error(f"�����ļ�Ŀ¼: {features_dir}")
                self.logger.error(f"Ŀ¼�Ƿ����: {features_dir.exists()}")
                if features_dir.exists():
                    existing_files = list(features_dir.glob("*.parquet"))
                    self.logger.error(f"Ŀ¼�е��ļ���: {len(existing_files)}")
                    if existing_files:
                        self.logger.error(f"ǰ5���ļ�: {[f.name for f in existing_files[:5]]}")
                return None

            self.logger.info(f"׼���ϲ����ݣ���{len(all_data)}���ļ�������")

            # �ϲ���������
            try:
                result_df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"���ݺϲ��ɹ����ܼ�¼��: {len(result_df)}")
            except Exception as e:
                self.logger.error(f"���ݺϲ�ʧ��: {e}")
                self.logger.error(f"all_data����: {len(all_data)}")
                for i, df in enumerate(all_data[:3]):  # ֻ���ǰ3��
                    self.logger.error(f"DataFrame {i}: shape={df.shape}, columns={len(df.columns)}")
                raise

            self.logger.info(f"�ɹ����ظ�������: {len(result_df)}����¼, {len(result_df.columns)}������")
            
            # ���浽����
            result_df.to_parquet(cache_file)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"���ظ�������ʧ��: {e}")
            return None
    

''')
if marker not in text:
    raise SystemExit('marker not found')
insert = dedent('''
    def load_sector_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载板块特征数据"""
        try:
            self.logger.info(f"加载板块特征数据: {start_date} - {end_date}")

            sector_data: Optional[pd.DataFrame] = None
            try:
                sector_data = self.base_loader.load_sector_features(start_date, end_date)
            except Exception as inner_exc:
                self.logger.warning(f"专业板块加载失败，将尝试简化逻辑: {inner_exc}")

            sector_data = self._post_process_sector_data(sector_data, start_date, end_date)

            if sector_data is None or sector_data.empty:
                self.logger.warning("板块数据为空或缺少关键字段，启用简化加载逻辑")
                sector_data = self._load_sector_features_basic(start_date, end_date)

            if sector_data is None or sector_data.empty:
                self.logger.error("板块数据加载失败")
                return None

            sector_count = (
                sector_data['sector_code'].nunique()
                if 'sector_code' in sector_data.columns else '未知'
            )
            self.logger.info(f"成功加载板块数据: {len(sector_data)} 条记录, {sector_count} 个板块")

            return sector_data

        except Exception as e:
            self.logger.error(f"板块数据加载失败: {e}")
            return None

    def _post_process_sector_data(
        self,
        data: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
        default_sector_code: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """保证板块数据具备因子计算所需字段"""
        if data is None:
            return None

        if data.empty:
            return data

        try:
            df = data.copy()
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if 'date' not in df.columns:
                if 'date_str' in df.columns:
                    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
                else:
                    self.logger.warning("板块数据缺少 date 字段，无法使用")
                    return None
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            df = df.dropna(subset=['date'])
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                return df

            if 'sector_code' not in df.columns:
                if 'sector' in df.columns:
                    df['sector_code'] = df['sector']
                elif 'sector_name' in df.columns:
                    df['sector_code'] = df['sector_name']
                elif default_sector_code is not None:
                    df['sector_code'] = default_sector_code
                else:
                    self.logger.warning("板块数据缺少 sector_code，且无法推断")
                    return None

            df['sector_code'] = df['sector_code'].astype(str)
            if 'sector_name' not in df.columns and 'sector' in df.columns:
                df['sector_name'] = df['sector']

            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            df = df.sort_values(['sector_code', 'date']).reset_index(drop=True)

            if 'pct_chg' in df.columns:
                df['return_1d'] = df['pct_chg'].astype(float)
            elif 'close' in df.columns:
                df['return_1d'] = df.groupby('sector_code')['close'].pct_change() * 100
            else:
                df['return_1d'] = 0.0

            df['return'] = df['return_1d']
            df['daily_return'] = df['return_1d']

            if 'close' in df.columns:
                group_close = df.groupby('sector_code')['close']
                for window in (3, 5, 10, 20):
                    col = f'return_{window}d'
                    if col not in df.columns:
                        df[col] = group_close.transform(lambda s, w=window: (s / s.shift(w) - 1.0) * 100)
            else:
                self.logger.warning("板块数据缺少 close 列，无法计算多日收益")

            for col in ['return_1d', 'return', 'daily_return', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
                if col in df.columns:
                    df[col] = df[col].astype(float).fillna(0.0)

            return df

        except Exception as exc:
            self.logger.warning(f"板块数据整理失败: {exc}")
            return None

    def _load_sector_features_basic(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """简化的板块数据加载逻辑，不强依赖额外衍生文件"""
        try:
            sector_root = system_config.sector_features_dir
            if not sector_root.exists():
                self.logger.warning(f"板块特征目录不存在: {sector_root}")
                return None

            frames: List[pd.DataFrame] = []
            for sector_dir in sector_root.iterdir():
                if not sector_dir.is_dir():
                    continue
                sector_code = sector_dir.name

                candidates = [
                    sector_dir / f"{sector_code}_features.parquet",
                    sector_dir / f"{sector_code}.parquet",
                ]
                feature_file = next((p for p in candidates if p.exists()), None)
                if feature_file is None:
                    continue

                try:
                    df = pd.read_parquet(feature_file)
                except Exception as read_exc:
                    self.logger.debug(f"读取板块文件失败 {feature_file}: {read_exc}")
                    continue

                processed = self._post_process_sector_data(df, start_date, end_date, default_sector_code=sector_code)
                if processed is not None and not processed.empty:
                    frames.append(processed)

            if not frames:
                self.logger.warning(f"简化加载仍未得到板块数据: {start_date} - {end_date}")
                return None

            sector_data = pd.concat(frames, ignore_index=True)
            sector_data = sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)
            self.logger.info(f"简化板块加载成功: {len(sector_data)} 条记录, {sector_data['sector_code'].nunique()} 个板块")
            return sector_data

        except Exception as exc:
            self.logger.error(f"简化加载板块数据失败: {exc}")
            return None

''')
text = text.replace(marker, marker + insert)
path.write_text(text, encoding="utf-8")
