# 任务：factors.parquet 新增 5 个纯行业因子列（SW2 广播口径）

> 交接口：本文件可直接转交 DeepSeek 执行。完成后 merge 引擎的 M16_ctx 腿池与
> mkt_sector_test 枚举评估即可引用这些列（merge 侧接线另行进行，本次只生成列）。

## 背景（必读）

- merge 引擎（`pandaai_online/merge/evaluator.py` 的 `ensure_local_fields`）已实现 SW2
  行业收益广播：`IND_RET = ret.groupby([日期, SW2行业]).transform('mean')`，广播到成员股。
  **新列必须与该口径逐位一致**（同日同行业所有成员股同值）——两引擎读数才可对照。
- SW2 映射加载：`config/sw2_industry_encoding.pkl` 的 `mappings.stock_to_industry`
  （参考 `pandaai_online/merge/field_catalog.py` 的 `sw2_mapping()`，进程内缓存写法可照抄）。
- 收益定义：`RET1 = close.pct_change(fill_method=None)`（停牌 → NaN，**不**前向填充；
  NaN 不计入行业均值/标准差）。与 merge 引擎 `ensure_local_fields` 的 ret 完全同源。
- 评审结论（2026-09-25）：固定 SW2 快照可接受（评估窗 2021-09-01 起在 2021-06-23
  版本更新之后；实测面板 5126 只股票 0 只未映射）；point-in-time 成分暂不做。

## 新增列清单（5 列，sector_ 前缀，广播到成员股）

| 列名 | 构造 | 含义 |
|---|---|---|
| `sector_mom_20` | TS_SUM(IND_RET, 20)（20 日行业收益累计） | 行业动量·短 |
| `sector_mom_60` | TS_SUM(IND_RET, 60) | 行业动量·中 |
| `sector_mom_term_5_120` | TS_MEAN(IND_RET,5) / TS_MEAN(IND_RET,120) − 1 | 行业动量期限结构（>0 = 短期强势）|
| `sector_vol_60` | TS_STD(IND_RET, 60) | 行业波动率 |
| `sector_disp_20` | TS_MEAN(行业内 RET1 截面标准差, 20) | 行业内涨跌分化（选股富矿环境）|

其中 IND_RET = 同日同行业 RET1 的均值（先算）；行业内分化 = 同日同行业 RET1 的
截面标准差（逐日先算，再做 20 日均值平滑——日频原始分化噪声大）。

## 构造要求

1. **数据源**：与 merge 引擎同源的 close（复权口径一致）；行业分组用
   `config/sw2_industry_encoding.pkl`（sw2_mapping，131 个行业，覆盖面板全部 5126 只）
2. **广播语义**：先构造（日期 × 行业）的行业时间序列，做 rolling 运算，再映射回
   成员股——同日同行业所有成员股必须同值（逐一校验，见验收 1）
3. **成员数下限**：当日同行业成员 <3 只 → 该组当日值置 NaN（小样本噪声；
   实测最小行业 1 只、p25=12 只，此规则主要影响尾部）
4. **rolling 语义**：min_periods=窗口大小（全窗），窗口未满输出 NaN（最长窗口 120 日，
   面板前段约半年 NaN 属预期）
5. **单位**：IND_RET 用小数收益（与 close.pct_change 一致），不要用百分比
6. **去重**：不要重复实现 `sector_momentum_ar1_20d` / `mkt_sector_rotation_speed_20d`
   （已存在，轮动速度/AR(1) 持续性已覆盖）；不要动市场级列（`mkt_vol_term_structure_5_20`
   等已存在）

## 验收标准

1. **广播校验**：任取一日，同行业成员股的每个新列值 nunique == 1（逐列逐日）
2. **覆盖率**：warmup 之后全面板非 NaN 比例 ≥95%
3. **层级识别**：`pandaai_online/merge/mkt_sector_test.py --factors <5 列逗号分隔>
   --label vwap` 能将全部 5 列判定为 sector 层（行业内常值），cs_spearman/cs_ir 非全 NaN
4. **口径对照**：`sector_mom_20` 与 merge 引擎 `TS_MEAN(IND_RET,20)`（evaluator
   ensure_local_fields 产的 IND_RET）的月末截面秩相关 |ρ| > 0.99（同口径验证）

## 不要做

- 不要动 merge 引擎代码（merge 侧已自行实现 IND_RET/DISP 本地列与 M16 腿池，
  两套 IND_RET 口径一致是本次的设计约束）
- 不要实现 point-in-time 行业成分（固定快照已评审通过；未来若接入
  tushare index_member_all 的 enter/out_date 再升级）
- 不要实现市场级新列（已存在）
- 不要用市值加权行业收益（等权口径与 merge 引擎一致是硬要求）
