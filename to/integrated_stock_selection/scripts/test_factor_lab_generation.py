#!/usr/bin/env python3
"""使用合成数据测试 factor_lab 因子生成流程。"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from factor_system.factor_lab import definitions as _factor_lab_definitions  # noqa: F401
    from factor_system.factor_lab.pipeline import FactorPipeline
    from factor_system.factor_lab.store import FactorDatasetStore
    from factor_system.factor_lab.commands import generate_and_store
except ImportError as exc:  # pragma: no cover - 真正运行时不应触发
    raise RuntimeError(f"无法导入 factor_lab 组件: {exc}")

TEST_STORE_PATH = PACKAGE_ROOT / "factor_lab_data" / "test_generation.parquet"


def _build_mock_stock_data() -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    records = []
    tickers = ["FAKE1", "FAKE2"]
    for code in tickers:
        for i in range(160):
            current_date = start + timedelta(days=i)
            pct_chg = float(np.sin(i / 10.0) * 2.0)
            records.append({
                'stock_code': code,
                'date': current_date,
                'pct_chg': pct_chg,
            })
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def main() -> None:
    stock_data = _build_mock_stock_data()

    pipeline = FactorPipeline()
    store = FactorDatasetStore(TEST_STORE_PATH)
    TEST_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = generate_and_store(
        factor_names=['returns_zscore'],
        data=stock_data,
        start_date='20240115',
        end_date='20240215',
        store=store,
        pipeline=pipeline,
        market_data=None,
        sector_data=None,
        context_extra=None,
    )

    generated = result.data
    print("测试因子生成完成")
    print(f"记录条数: {len(generated)}")
    print(f"时间范围: {generated['date'].min()} -> {generated['date'].max()}")
    print(f"存储文件: {TEST_STORE_PATH.resolve()}")


if __name__ == '__main__':
    main()
