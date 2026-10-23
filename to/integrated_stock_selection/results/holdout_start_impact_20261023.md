# HOLDOUT_START 对 84 列 LGB 排序的影响

## 结论

**没有影响。`build_top.py` 的 84 列 LGB 排序不调用 `HOLDOUT_START`，也不使用 train/holdout 切分。**

## 证据链

1. `build_top.py` 只导入：
   ```python
   from fitness import DataContext, build_label_series, sia_of, tier1
   ```
   没有导入 `HOLDOUT_START`，也没有调用 `train_holdout` / `split_mask`。

2. `build_top.py` 的 `DataContext` 构造只传 `end`：
   ```python
   ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev,
                            end=(end or W.DEFAULT_DATA_CUT))
           for lab in ('c2c', 'vwap')}
   ```
   没有传 `cut`。

3. `eval_mined` / `eval_custom` / `eval_store` 使用：
   ```python
   m = ctx.screen(v, name=...)
   lgb = lgb_score(m)
   ```
   `ctx.screen(..., mask=None)` 使用整个 `DataContext` 窗口；`tier1(m)` 也不做 train/holdout 切分；`refs.lgb_score(m)` 直接使用全窗 `rank_ic_mean`、`rank_ic_ir`、`rank_ic_win_rate`、`monotonicity`、`turnover`。

4. `build_top` 的实际窗口：
   ```python
   _fit_end = min(str(FIT_END), str(args.data_cut))
   ```
   默认 `data_cut=2025-12-31`，所以 LGB 排序窗为 **2021-09-01 ~ 2025-12-31**，不是 2025-07-01 后的 holdout。

## 影响范围

`HOLDOUT_START=20250701` 只影响：
- `fitness.train_holdout` / `run_search` 的 train-vs-holdout 诊断；
- 不影响 `build_top` 的 `LGB` 排序、84 列选择或 2025-12-31 前的全窗指标。

因此，84 列 LGB 排序不存在“被 2025-07-01 holdout 切分改变”的问题；真正需要关注的是 **mined 因子的公式发现使用了 2026-08-21 数据**（见 `results/check1_time_boundary_20261023/summary.md`）。
