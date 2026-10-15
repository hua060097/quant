# -*- coding: utf-8 -*-
"""自定义 Alpha101 与论文因子示例库（在此文件或同目录新建 .py 即可自动注册）。"""
import numpy as np
import pandas as pd
from custom_registry import register_formula_factor, register_py_factor

# 方式 1：直接用平台/本地 DSL 公式注册 Alpha101
register_formula_factor(
    factor_id="alpha101_006",
    formula="0 - CORR(OPEN, VOLUME, 10)",
    axis="pv_relation",
    description="Alpha#6: -1 * correlation(open, volume, 10)"
)

register_formula_factor(
    factor_id="alpha101_012",
    formula="SIGN(DIFF(VOLUME, 1)) * (0 - DIFF(CLOSE, 1))",
    axis="pv_relation",
    description="Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"
)


# 方式 2：用原生 Python / Pandas 编写复杂论文因子或带截面中性化的 Alpha101
@register_py_factor(
    factor_id="paper_idio_mom_20d",
    axis="trend_position",
    description="剥离市场Beta与行业收益后的20日特质收益风险调整动量"
)
def paper_idio_mom_20d(frame: pd.DataFrame, ev) -> pd.Series:
    codes = frame['stock_code']
    # 个股超额行业且剥离市场 Beta 后的日残差（前向窗口，无未来函数）
    ret1 = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    resid = ret1 - frame['IND_RET'] - frame['BETA20'] * (frame['MKT_RET'] - frame['IND_RET'])
    g = resid.groupby(codes, sort=False)
    mean20 = g.transform(lambda s: s.rolling(20, min_periods=15).mean())
    std20 = g.transform(lambda s: s.rolling(20, min_periods=15).std())
    return mean20 / std20.where(std20 > 1e-8)
