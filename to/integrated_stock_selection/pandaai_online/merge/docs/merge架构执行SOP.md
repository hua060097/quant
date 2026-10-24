# Merge 因子搜索与特征矩阵构建标准执行手册 (SOP)

## 0. 执行前硬性代码检查（必做）

1. **质量门语义**：`build_top.py::_quality_scan_positions` 默认是 `top + quality` 并集，不是交集；`--full-scan` 仍表示全表。
   V6 要“只扫过线因子”时，必须显式传 **`--quality-only`**（与 `--full-scan` 配合）。
2. **父进程预热**：`run_search.py` 与 `build_top.py` 必须在 fork 前调用
   `ensure_extended_fields(frame)` 与 `ensure_gate_fields(frame)`，避免子进程重复物化扩展/门控列。
3. **放行 fast seats**：`build_top.py::_apply_incremental_guard` 必须同时过滤
   `fast_seats/fast_detail`，不能只过滤 picked/pool/seats/habitat。
4. **台账隔离**：V6 建议使用新台账
   `results/merge_v2/v6_clean/ledger_v6.jsonl`，旧 `ledger.jsonl` 保留不删。

---

## Step 1：V4.3d GP 因子搜索

> 严格 `data_cut=2025-12-31`；`--ledger` 与 `--data-cut` 是全局参数，必须放在 `search` 之前。
> `--gate-focus` 会在基础 seed 之外追加含 `MKT_GATE1/2`、`SEC_GATE1/2` 的 M14/M16 种子。

```bash
python3 -B -u pandaai_online/merge/run_search.py \
  --ledger results/merge_v2/v6_clean/ledger_v6.jsonl \
  --data-cut 2025-12-31 \
  search \
  --strategy mechanism \
  --n 2000 \
  --gate-focus 0.5 \
  --refine-children 2 \
  --full-cap 120 \
  --top 40 \
  --include-custom \
  --jobs auto
```

注意：
- 当前合法的细调参数是 `--refine-children`，不是 `--refine-rounds`。
- `--jobs auto` 在 POSIX 使用 fork 进程池；Windows 自动退化为串行。
- 如果只做快速 smoke，建议加 `--fast-stocks 100 --fast-months 6 --top 1 --full-cap 2`。

---

## Step 2：全量过线因子评估与正交去重

> 默认 `top+quality` 不再足够；V6 必须用 `--full-scan --quality-only`。
> `--quality-only` 只返回同时满足 `|RankIC| >= min_abs_rankic` 且 `LGB >= min_lgb` 的行。

```bash
python3 -B -u pandaai_online/merge/build_top.py \
  --top 2000 \
  --full-scan \
  --quality-only \
  --min-abs-rankic 0.012 \
  --min-lgb 0.0015 \
  --keep-pool 200 \
  --keep-base 20 \
  --habitat-seats 10 \
  --habitat-pool 10 \
  --habitat-max-extra 0 \
  --include-custom \
  --include-store \
  --export-panel \
  --data-cut 2025-12-31 \
  --jobs auto
```

关键语义：
- `--full-scan` 单独使用时仍是全表；只有叠加 `--quality-only` 才是“全表质量门”。
- `--keep-pool 200` 是结果池上限，不等于“扫描上限”。
- `--habitat-max-extra 0` 表示不做额外截断。
- `--export-panel` 才能落盘 `vwap_top20_factors.parquet` / `vwap_habitat_pool.parquet`。

---

## Step 3：偏科因子配对、门控交互与 F1/FDR

### 3.1 Per-Gate Quota 配对

> 每个认证 market gate 至少 2 个、最多 5 个候选；gain/flip 总上限 35，switch 另算。

```bash
python3 -B -u pandaai_online/merge/select_specialists.py \
  --data-cut 2025-12-31 \
  --bank results/merge_v2/gates_search/certified_gate_bank.json \
  --per-gate-min 2 \
  --per-gate-max 5 \
  --max-market-cands 35 \
  --jobs 16
```

实际 active bank gate 只有 7 个有 selected 候选时，gain 约为 28 个，加 switch 4 个，合计约 32 个。

> `--bank` 必须是本次 V4.3d 对应的 **bank-v2**（10 个市场门控）。
> 本地主 `certified_gate_bank.json` 可能是旧 bank-v1（3 个市场门控）；先确认/备份，否则 F1 会按错 bank 过滤。

### 3.2 F1 批量检验与 FDR

```bash
python3 -B -u pandaai_online/merge/f1_batch.py \
  --cands results/merge_v2/specialist_cands_gain.json \
  --out results/merge_v2/gates_search/f1_specialist_gain.csv \
  --form gain --base-panel specialist \
  --gate-bank results/merge_v2/gates_search/certified_gate_bank.json \
  --label-max 2025-12-31 --jobs 16

python3 -B -u pandaai_online/merge/f1_batch.py \
  --cands results/merge_v2/specialist_cands_switch.json \
  --out results/merge_v2/gates_search/f1_specialist_switch.csv \
  --form switch --base-panel specialist \
  --gate-bank results/merge_v2/gates_search/certified_gate_bank.json \
  --label-max 2025-12-31 --jobs 16

python3 -B -u pandaai_online/merge/judge_f1.py \
  --family results/merge_v2/gates_search/family_manifest_F1v3_gain.json \
  results/merge_v2/gates_search/f1_specialist_gain.csv

python3 -B -u pandaai_online/merge/judge_f1.py \
  --family results/merge_v2/gates_search/family_manifest_F1v3_switch.json \
  results/merge_v2/gates_search/f1_specialist_switch.csv
```

### 3.3 实例化 Class 4 交互列

`materialize_interactions.py` 每次只接受一个 `--judged`，**不要用同一个 `--out` 连续覆盖**。先分别产出，再按键合并：

```bash
python3 -B -u pandaai_online/merge/materialize_interactions.py \
  --judged results/merge_v2/gates_search/f1_specialist_gain_judged.csv \
  --cands results/merge_v2/specialist_cands_gain.json \
  --base-panel specialist \
  --out results/merge_v2/approved_interactions_gain.parquet

python3 -B -u pandaai_online/merge/materialize_interactions.py \
  --judged results/merge_v2/gates_search/f1_specialist_switch_judged.csv \
  --cands results/merge_v2/specialist_cands_switch.json \
  --base-panel specialist \
  --out results/merge_v2/approved_interactions_switch.parquet

python3 - <<'PY'
import pandas as pd
a = pd.read_parquet('results/merge_v2/approved_interactions_gain.parquet')
b = pd.read_parquet('results/merge_v2/approved_interactions_switch.parquet')
c = a.merge(b, on=['stock_code', 'date'], how='outer')
c.to_parquet('results/merge_v2/approved_interactions.parquet', index=False)
print(c.shape)
PY
```

行业 switch 继续走 `f1_sector.py` + 对应 judge/materialize，再按同样方式合并；不要只跑 market 就导出 Class 4。

---

## Step 4：导出特征矩阵与 2026 延伸

> V6 显式开启 interact 相关性去重与 Class1 家族/主字段均衡。
> 当前 `--max-family-frac` 会把 `family=store` 细分为 `store:<roles.csv group>`。

```bash
python3 -B -u pandaai_online/merge/export_lgb_dataset.py \
  --data-cut 2025-12-31 \
  --max-alpha 200 \
  --max-interact 50 \
  --max-sector 2 \
  --cs-norm mad \
  --interaction-rho-max 0.70 \
  --max-family-frac 0.30 \
  --max-field-frac 0.30 \
  --diversity-report results/merge_v2/lgb_ready_v6/diversity_report.json \
  --out-dir results/merge_v2/lgb_ready_v6

python3 -B -u pandaai_online/merge/export_2026_extension.py \
  --data-cut 2026-09-18 \
  --extend-from 2025-12-31 \
  --out-dir results/merge_v2/lgb_ready_v6
```

导出后必须检查：
- `features_manifest.json` 的 class1/class4 数量；
- `diversity_report.json` 的 family/field counts 与被裁剪列；
- `lgb_features.parquet` 的非 NaN 覆盖率。

本 SOP 只覆盖 merge 搜索到矩阵导出；Stage15 11 折 A/B 不在本 SOP 内，必须另跑同 fold 表对照。
