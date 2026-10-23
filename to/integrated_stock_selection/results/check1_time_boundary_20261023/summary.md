# Check 1: run_search / fitness 时间边界

## 结论

**挖矿候选链包含 2026 年数据；84 的最终 build_top 筛选不包含 2026 年。**

- `stock_features` 文件本身：`cache/stock_features_20200101_20260918.parquet`，最晚 **2026-09-18**。
- `run_search.py` 的 `load_panel()` 读完整面板；`DataContext` 再按 `self.data_cut` 截断（`run_search.py:344-345`、`:413`）。
- `fitness.py:29-31` 硬编码：
  - `START='20210901'`
  - `END='20260821'`
  - `HOLDOUT_START='20250701'`
- `ledger.jsonl` 15,551 条事件：
  - `data_cut=2026-08-21`: **15,329 条**（全部 `data_cut_legacy=true`，属事后回填标记）
  - `data_cut=2025-12-31`: **222 条**
  - 事件时间：2026-09-15 ~ 2026-10-15
- 因此挖矿候选的**实际评估窗上界是 2026-08-21**；vwap 标签使用 T+2/T+3，信号 2026-08-21 的标签还会读到 2026-08-26 的价格。**不是 2025-12-31。**

## 84 的最终筛选

`build_top.py` 实际：
```python
_fit_end = min(str(FIT_END), str(args.data_cut))
```
默认 `args.data_cut = W.DEFAULT_DATA_CUT = '2025-12-31'`，所以：
```text
min('20260821', '2025-12-31') = 2025-12-31
```
即 **build_top 的筛选窗是 2021-09-01 ~ 2025-12-31**；2026 数据没有进入 84 的最终排名。`features_manifest.json` 也标记 `data_cut=2025-12-31`。

## mined 因子数量

按 `results/for_gemini_20261020/merge_v2_lgb_ready_extension_manifest_2026.json` 的 `sources`：

- `mined`: **24 个**（class1 20 个 + habitat pool 4 个）
- `store`: 28
- refs: 3
- custom: 1

所以实际不是 19 个；若之前的 19 是旧口径，核心结论不变：**这些 mined 因子的发现/候选证据使用了 2026 数据，84 只是用 2025 窗口重新排名。**

## 影响

- 不是“84 的 CV 指标偷看 2026”；84 的 CV 仍截止 2025-12-31。
- 但 **mined 因子的公式选择/候选生成**使用了 2026-01~08 的信息，属于**发现阶段的未来信息污染**。
- 若要求严格 PIT，需用 `data_cut=2025-12-31` 重新生成 mined 候选，或把现有 mined 因子标为 legacy，不应与纯 2025 前挖掘因子混为一谈。
