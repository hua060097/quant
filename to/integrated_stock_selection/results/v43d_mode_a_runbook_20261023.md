# V4.3d 模式 A 严格重建 Runbook（本地未执行全量）

> 目标：严格 `data_cut=2025-12-31`，跑 2000 基础候选 + `--gate-focus 0.5`（约 3000 个总候选），再做 build_top 与 stage15 A/B。
> 当前本地没有 2026 训练矩阵/服务器，未启动全量。

## 0. 保护现场

```powershell
$ts = Get-Date -Format yyyyMMdd_HHmmss
Copy-Item pandaai_online/merge/ledger.jsonl "pandaai_online/merge/ledger_backup_$ts.jsonl"
New-Item -ItemType Directory -Force results/merge_v2/v6_clean | Out-Null
Copy-Item results/merge_v2/gates_search/certified_gate_bank.json "results/merge_v2/v6_clean/certified_gate_bank_v1_backup.json"
```

模式 A 需要 bank-v2（10 个市场门控）。确认后把 server bank-v2 放到 `results/merge_v2/gates_search/certified_gate_bank.json`。

## 1. V4.3d GP 搜索（严格 2025-12-31）

在 Linux 24C/185G 服务器：

```bash
python pandaai_online/merge/run_search.py \
  --ledger results/merge_v2/v6_clean/ledger_v6.jsonl \
  --data-cut 2025-12-31 \
  search \
  --strategy random \
  --n 2000 \
  --gate-focus 0.5 \
  --jobs auto \
  --top 40 \
  --full-cap 100 \
  --include-custom \
  --local-context \
  --allow-documented \
  --seed 43 \
  --report
```

- `--n 2000` + `--gate-focus 0.5` 目标约 3000 个候选。
- `--gate-focus` 只生成 M14/M16 中含 `MKT_GATE1/2`、`SEC_GATE1/2` 的额外种子。
- 慢路径由 `--jobs auto` 走 POSIX fork `ProcessPoolExecutor`；Windows 自动退化串行。

## 2. 新矩阵 + 特征清单

```bash
python pandaai_online/merge/build_top.py \
  --data-cut 2025-12-31 --top 200 --jobs auto

python pandaai_online/merge/export_lgb_dataset.py \
  --data-cut 2025-12-31 \
  --out-dir results/merge_v2/v6_clean/lgb_ready
```

生成 V6 特征清单后，转成 stage15 的 `--features_file` 格式：

```bash
python scripts/make_stage15_features_file.py \
  --manifest results/from_server_20261020/features_manifest.json \
  --out results/v6_clean/features84_baseline.json

python scripts/make_stage15_features_file.py \
  --manifest results/merge_v2/v6_clean/lgb_ready/features_manifest.json \
  --out results/v6_clean/features_v6.json
```

## 3. Stage15 A/B（严格 CV）

```bash
python main_alpha_stage15_hp_search.py \
  --mode all --cv_end 2025-12-31 \
  --features_file results/v6_clean/features84_baseline.json \
  --quick_dir_out results/stage15hp_quick_baseline \
  --full_dir_out results/stage15hp_full_baseline

python main_alpha_stage15_hp_search.py \
  --mode all --cv_end 2025-12-31 \
  --features_file results/v6_clean/features_v6.json \
  --quick_dir_out results/stage15hp_quick_v6 \
  --full_dir_out results/stage15hp_full_v6
```

- `--cv_end 2025-12-31` 是新增参数；默认仍是 2026-03-01，避免影响旧生产。
- A/B 必须使用同一 fold 表、同一超参网格、同一随机种子；否则不是受控对照。
- 若 stage15 实际折数不是 11，以生成的 fold 表为准，不要宣称 11 折。
