# -*- coding: utf-8 -*-
"""
Stage1.5 HP 搜索：两阶段公平超参搜索（针对当前 CURATED 因子集）

背景：P06 是 Stage1 在 110 因子集上从 P01-P08 中按"硬门槛+median最高"选出的冠军。
      后续加因子仍用 P06 固定超参对比不公平——新因子集应先搜超参再对比。

阶段A（quick）：
  全网格超参 × 全部折(默认 Fold1~6) × 600 轮(与 full 同轮数，落盘全量 cp 供 full 复用)
  × checkpoint [200,300,350,400,450,500,600]，LightGBM num_threads=3（单进程防OOM）
  [EXPAND 2026-09-23 用户指令] 冠军候选 cp 扩为全 7 档、折集扩为全 6 折 → 一次更全面的搜索
  （基本可替代 full；full 逻辑保留）。
  [CHAMP-RULE 2026-09-24] 冠军选择规则默认改为 min+mean（0.5×min_fold_mean+0.5×mean，折间一致性+右尾感知），
  可选 --champion_rule median|mean|min_fold_mean|pos_folds|slot2_sharpe|slot3_sharpe|min+mean+slot2；
  median 口径冠军一并记录进 champion.json 供对照。
  [SLOT-SIM 2026-09-25] quick 汇总附加日频槽位组合净值（2/3槽、真实可实现口径，成本0.16%往返），
  部署级验证：同窗6配置的2槽排序与 rqalpha 分钟引擎完全一致。
  另统计 Rank1~5 均值/中位数单调性与 Top5 均收益，仅打印/落盘辅助判断，不参与记分。
  输出 hp_rank_daily.csv / hp_stock_details_top5.csv / hp_feature_importance.csv
      / hp_summary.csv / hp_top5_summary.csv / hp_config.json / champion.json / run_quick.log
  选择规则：MDD_min>-60% 且 q10>-10% 门槛内 → 按 --champion_rule 排序最高 → 冠军超参

阶段B（full）：
  冠军超参 × 全部折 × 600 轮 × checkpoint [200,300,350,400,450,500,600]
  LightGBM num_threads=FULL_NUM_THREADS（默认3，与 quick 同速；deterministic=True
  已两次实测跨线程数逐bit一致 → 与原 110 基线口径可比性不变）
  [REUSE 2026-09-22] quick 已跑过的折直接复用其落盘行（data_fp + 超参定义一致时），
  只重训剩余折（实测 quick/full 公共折×公共cp 跨线程数逐bit一致）
  输出与 main_alpha_stage15_rank_diag.py 完全一致的 rank_daily.csv /
      stock_details_top5.csv / feature_importance.csv / config.json
  → 直接 compare_stage15.py --new 对比

用法：
  python main_alpha_stage15_hp_search.py                     # quick + full 全流程
  python main_alpha_stage15_hp_search.py --mode quick        # 只跑快速搜索
  python main_alpha_stage15_hp_search.py --mode full --quick_dir results\\stage15hp_quick_xxx  # 单独跑全量验证
  python main_alpha_stage15_hp_search.py --mode full --force_pid P06  # 指定超参跑全量
  python main_alpha_stage15_hp_search.py --train_months 12   # 覆盖训练窗口（默认18）
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
MAIN_SRC = _HERE / "main_alpha_二分类_原因子_with_test_metrics_妖股_copy_原始_20260720_regression_rankgauss_binary_labelround.py"

EXP_TAG = "STAGE15-HP-SEARCH"
DATA_START = "2023-06-01"
# [FIX 2026-09-08] 加载止点改自动探测（曾硬编码2026-08-10致09月数据静默不可用）：
# 取 min(因子库最大日, 本地交易日历最大日, 今天)，日志打印实际值。None=自动。
DATA_LOAD_END = None
CV_END = "2026-03-01"
TRAIN_MONTHS = 18
TEST_MONTHS = 2
EMBARGO_TRADE_DAYS = 5

# 阶段A：快速搜索配置
QUICK_ROUNDS = 500
# [EXPAND 2026-09-23 用户指令] 冠军候选 cp 扩为全 7 档（原 [200,300,400,500]）；折集扩为全 6 折（原 [1,2,4,5,6]）。
#   目的：一次更全面的搜索（基本替代 full，full 逻辑保留），检验 deploy 崩是否由超参/cp 选择导致。
QUICK_CHECKPOINTS = [200, 300, 350, 400, 450, 500, 600]
QUICK_NUM_THREADS = 3
QUICK_FOLD_IDS = [1, 2, 3, 4, 5, 6]

# 阶段B：全量验证配置（与 110 基线 stage15 完全一致）
FULL_ROUNDS = 600
FULL_CHECKPOINTS = [200, 300, 350, 400, 450, 500, 600]

# [SPEED 2026-09-22] full 线程数 = 与 quick 相同。原 full 用 n_jobs=1 是为"与110基线逐bit同口径"，
#   但 LightGBM deterministic=True 跨线程数稳定已两次实测（quick 3线程 vs full 1线程，公共折×cp
#   2520/2520、3360/3360 逐bit相同；quick→full 复用机制也基于该结论）→ full 多线程不影响可比性。
FULL_NUM_THREADS = 3

# [REUSE 2026-09-22] quick 训练轮数/cp 与 full 对齐，使 full 可直接复用 quick 落盘行：
#   实测 quick(num_threads=3) 与 full(n_jobs=1) 同轮数下公共折×公共cp 逐bit一致
#   （LightGBM deterministic=True 跨线程数稳定，且高轮模型的前 N 棵树与总轮数无关），
#   因此 full 在 data_fp + 超参定义全等时复用 quick 行是安全的（详见 _quick_reuse_ok）。
QUICK_TRAIN_ROUNDS = max(QUICK_ROUNDS, FULL_ROUNDS)
QUICK_ALL_CHECKPOINTS = sorted(set(FULL_CHECKPOINTS) | set(QUICK_CHECKPOINTS))

# 选择规则（与 compare_stage15.py 同口径）
GATE_MDD_MAX = -0.60
GATE_Q10_MIN = -0.10

# 13 组超参 (2026-09-07 用户重写全网格：新增H9/H10/H11/H12/H13，spw 20~90成新轴；
#   注意pid lineage已断（H2~H8规格全变），历史H结论不再直接可比，对比以full spec为准)
#   H2保留(111冠军, 112下仍稳) / 其余向中心收敛：过重正则/过极采样/过高min_leaf均回调
#   H6更换理由：三轮同协议快搜36格中均值-0.0030垫底、12格仅1正(H6@500@C=+0.0033)，leaves63小树范式无赢面
# HP_GRID = [
#     {"pid": "H2", "num_leaves": 127, "min_data_in_leaf": 1200, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},  # 保留：111冠军，112下仍稳
#     {"pid": "H3", "num_leaves": 63,  "min_data_in_leaf": 1500, "feature_fraction": 0.6, "bagging_fraction": 0.7, "reg_alpha": 20, "reg_lambda": 30, "scale_pos_weight": 48.0},  # 优化：重正则回调 2000→1500,0.7→0.6,30/50→20/30，防过拟合不足
#     {"pid": "H4", "num_leaves": 127, "min_data_in_leaf": 800,  "feature_fraction": 0.65,"bagging_fraction": 0.6, "reg_alpha": 12, "reg_lambda": 12, "scale_pos_weight": 48.0},  # 优化：600→800,0.7→0.65,0.5→0.6,10/10→12/12 向中心
#     {"pid": "H5", "num_leaves": 127, "min_data_in_leaf": 1200, "feature_fraction": 0.45,"bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},  # 优化：ff 0.4→0.45 极致回调
#     {"pid": "H6", "num_leaves": 127, "min_data_in_leaf": 1000, "feature_fraction": 0.6, "bagging_fraction": 0.65,"reg_alpha": 12, "reg_lambda": 10, "scale_pos_weight": 48.0},  # 更换(2026-09-05)：leaves63→127，向H1/H4中心收敛，探1000/0.6/12/10空位
#     {"pid": "H7", "num_leaves": 127, "min_data_in_leaf": 1300, "feature_fraction": 0.75,"bagging_fraction": 0.7, "reg_alpha": 12, "reg_lambda": 15, "scale_pos_weight": 48.0},  # 优化：1500→1300,0.8→0.75,10/20→12/15
#     {"pid": "H8", "num_leaves": 127, "min_data_in_leaf": 2000, "feature_fraction": 0.7, "bagging_fraction": 0.75,"reg_alpha": 25, "reg_lambda": 40, "scale_pos_weight": 48.0, "max_depth": 7},  # 优化：2400→2000,0.8→0.75,30/50→25/40 轻回调
#     {"pid": "H1", "num_leaves": 127, "min_data_in_leaf": 800,  "feature_fraction": 0.55,"bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 7,  "scale_pos_weight": 48.0},  # 优化：600→800,0.5→0.55,5→7
#     {"pid": "H9", "num_leaves": 31, "min_data_in_leaf": 1300, "feature_fraction": 0.5, "bagging_fraction": 0.4, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},  # 保留：111冠军，112下仍稳
#     {"pid": "H10", "num_leaves": 127, "min_data_in_leaf": 2000, "feature_fraction": 0.7, "bagging_fraction": 0.75,"reg_alpha": 25, "reg_lambda": 40, "scale_pos_weight": 30.0, "max_depth": 7},  # 优化：2400→2000,0.8→0.75,30/50→25/40 轻回调
#     {"pid": "H11", "num_leaves": 63,  "min_data_in_leaf": 1500, "feature_fraction": 0.6, "bagging_fraction": 0.5, "reg_alpha": 20, "reg_lambda": 30, "scale_pos_weight": 48.0},  # 优化：重正则回调 2000→1500,0.7→0.6,30/50→20/30，防过拟合不足


# ]
HP_GRID = [
    # [ABLATION 结论 2026-09-21 12:14] hardneg=off 消融（H9/H12 × 3折 × 同种子同快照）：
    #   H9@500 mean 0.0165→0.0375 / median 0.0116→0.0209 / win 0.556→0.579；
    #   H12@500 mean 0.0228→0.0376 / median 0.0114→0.0301 / win 0.556→0.611；
    #   代价：q10 -0.05→-0.08（仍在 GATE_Q10_MIN=-0.10 内）、mdd_mean 变差。
    #   判定：新 50 因子池下 hardneg 不再有保护作用（其 shape 组概念已随特征池更换失效），
    #   维持 CS_HARDNEG_ENABLE=False（_load_main 覆盖，主文件未动）。
    # {"pid": "H2", "num_leaves": 127, "min_data_in_leaf": 1700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 40.0},  # 保留：111冠军，112下仍稳  临时
    # {"pid": "H3", "num_leaves": 127, "min_data_in_leaf": 1700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 80.0},  # 保留：111冠军，112下仍稳
    # {"pid": "H4", "num_leaves": 127, "min_data_in_leaf": 700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 40.0},  # 保留：111冠军，112下仍稳
    # {"pid": "H5", "num_leaves": 127, "min_data_in_leaf": 2700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 40.0},  # 保留：111冠军，112下仍稳
    # {"pid": "H6", "num_leaves": 127, "min_data_in_leaf": 600, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 8, "reg_lambda": 8,  "scale_pos_weight": 30.0},  # 保留：111冠军，112下仍稳
    # [PRUNE 2026-09-28] H3 删：6轮统计 best_med 最高+0.0025、mean_med 全负；leaves63范式无赢面（H11同，H6/H7/H10已有先例）；省quick成本
    # {"pid": "H3", "num_leaves": 63,  "min_data_in_leaf": 1500, "feature_fraction": 0.6, "bagging_fraction": 0.7, "reg_alpha": 20, "reg_lambda": 30, "scale_pos_weight": 90.0},  # 优化：重正则回调 2000→1500,0.7→0.6,30/50→20/30，防过拟合不足
    # {"pid": "H4", "num_leaves": 63, "min_data_in_leaf": 800,  "feature_fraction": 0.65,"bagging_fraction": 0.6, "reg_alpha": 12, "reg_lambda": 12, "scale_pos_weight": 30.0},  # 优化：600→800,0.7→0.65,0.5→0.6,10/10→12/12 向中心  临时
    # {"pid": "H5", "num_leaves": 127, "min_data_in_leaf": 1200, "feature_fraction": 0.45,"bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 30.0},  # 优化：ff 0.4→0.45 极致回调  临时
    # # [PRUNE 2026-09-17] H6 0胜/17轮 best_med 0.0054(垫底档) → 删，省quick成本
    # {"pid": "H6", "num_leaves": 127, "min_data_in_leaf": 1000, "feature_fraction": 0.3, "bagging_fraction": 0.65,"reg_alpha": 12, "reg_lambda": 10, "scale_pos_weight": 30.0},  # 更换(2026-09-05)：leaves63→127，向H1/H4中心收敛，探1000/0.6/12/10空位
    # [PRUNE 2026-09-17] H7 0胜/16轮 best_med 0.0030(最低)+mean_med -0.0040(最低) → 删
    # {"pid": "H7", "num_leaves": 127, "min_data_in_leaf": 1300, "feature_fraction": 0.75,"bagging_fraction": 0.7, "reg_alpha": 12, "reg_lambda": 15, "scale_pos_weight": 30.0},  # 优化：1500→1300,0.8→0.75,10/20→12/15
    {"pid": "H8", "num_leaves": 127, "min_data_in_leaf": 2000, "feature_fraction": 0.7, "bagging_fraction": 0.75,"reg_alpha": 6, "reg_lambda": 12, "scale_pos_weight": 70.0, "max_depth": 7},  # 优化：2400→2000,0.8→0.75,30/50→25/40 轻回调
    # {"pid": "H1", "num_leaves": 127, "min_data_in_leaf": 800,  "feature_fraction": 0.55,"bagging_fraction": 0.8, "reg_alpha": 10, "reg_lambda": 7,  "scale_pos_weight": 25.0},  # 优化：600→800,0.5→0.55,5→7   临时
    # {"pid": "H9", "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 20.0},  # 保留：111冠军，112下仍稳     临时
    # {"pid": "H13", "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 80.0},  # [FIX 2026-09-07] 原pid误写H2与第86行重名致quick混行，已改H13；spw=80探高权重端    临时
    {"pid": "H12", "num_leaves": 31, "min_data_in_leaf": 1200, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 30.0},  # 保留：111冠军，112下仍稳；[PRUNE 2026-09-17] 0胜但mean_med +0.0004(第2)作leaves31+min1200多样性角保留
    # [HP-EXPAND 2026-09-28] H15 = H12 单变量：仅 spw 30→48，其余逐项相同。
    #   依据：leaves127 下 spw48+(H14/H4)显著优于 spw40(H2/H5)，spw 是成功组与失败组差异最大的单轴；
    #   H12 的 spw=30 是全网最低，测 pos_weight 在 leaves31 regime 是否成立（H3-spw90 反例在先，
    #   spw 效应与 regime 交互，故做实验而非直接认定）。备选轴 min_leaf→2000 证据弱一档（H12 的 1200 未输过），列后。
    {"pid": "H15", "num_leaves": 31, "min_data_in_leaf": 1200, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},
    # [HP-EXP 2026-09-30] H15S = H15 单变量：num_leaves 31→15（浅树容量正则；论文锚
    #   GKX 2020 GBRT depth2-3）。目的：2×2 factorial（池71/228 × 叶31/15）检验
    #   "228全入池在31叶下崩、浅树正则下是否成立"。其余参数与 H15 逐项相同（单轴原则）。
    {"pid": "H15S", "num_leaves": 15, "min_data_in_leaf": 1200, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},
    # [HP-EXPAND 2026-09-29] H16 = 干净的63叶：继承 H12/H15 的轻正则(10/5)+低min_leaf(1200)+ff/bf(0.5/0.6)，
    #   与已删 H3/H11 的区别：min_leaf 1500→1200、reg 20/30→10/5（旧63叶死于重正则+高min_leaf捆绑，
    #   不是叶子数本身）。spw 取 48（与 H15 对齐；若 H16 偏 H8 侧说明容量主导峰，若偏 H12 侧说明叶子数主导谷）。
    # [PRUNE 2026-09-29] H16 删除：R1 七格全灭（mean_avg -0.0008 全组最低、minfold_avg -0.0207 最深、
    #   无一格可取）。结论：63叶在轻正则/低min_leaf下是"峰谷双输"的死亡区，不是中间甜蜜点；
    #   叶子数轴非单调，31叶路线是唯一出路。省quick成本。
    # {"pid": "H16", "num_leaves": 63, "min_data_in_leaf": 1200, "feature_fraction": 0.5, "bagging_fraction": 0.6, "reg_alpha": 10, "reg_lambda": 5,  "scale_pos_weight": 48.0},
    # [PRUNE 2026-09-17] H10 0胜/16轮 best_med 0.0043(倒二) → 删
    # {"pid": "H10", "num_leaves": 127, "min_data_in_leaf": 2000, "feature_fraction": 0.4, "bagging_fraction": 0.75,"reg_alpha": 25, "reg_lambda": 40, "scale_pos_weight": 30.0, "max_depth": 7},  # 优化：2400→2000,0.8→0.75,30/50→25/40 轻回调
    # {"pid": "H11", "num_leaves": 63,  "min_data_in_leaf": 1500, "feature_fraction": 0.6, "bagging_fraction": 0.5, "reg_alpha": 20, "reg_lambda": 30, "scale_pos_weight": 48.0},  # 优化：重正则回调 2000→1500,0.7→0.6,30/50→20/30，防过拟合不足   临时
    # [HP-EXPAND 2026-09-23 用户指令] 补5组（总8组：H3/H8/H12 保留 + 以下5组），
    #   目的：检验"deploy 崩是否由超参/cp 选择导致"——新增锚点=历史 deploy 同款规格（旧池时代 OOS 为正者），
    #   与现 3 组同池同折对照。H8(6/12/70) 原样保留作失败锚点。
    # [PRUNE 2026-09-28] H2 删：6轮 best_med 最高+0.0021、mean_med 全负；min_leaf1700轴无赢面；与H4/H5同属保守配置簇(ff≤0.65/无max_depth/低spw)，合并无意义，直接删
    # {"pid": "H2", "num_leaves": 127, "min_data_in_leaf": 1700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 40.0},   # [旧池deploy最佳] H2_cp350 早期段 cum+498%/mdd-0.20（FINAL-DEPLOY同款规格）
    # [PRUNE 2026-09-28] H11 删：6轮 best_med 最高+0.0009、mean_med 全负（8组最差）；leaves63范式无赢面
    # {"pid": "H11", "num_leaves": 63,  "min_data_in_leaf": 1500, "feature_fraction": 0.6, "bagging_fraction": 0.5, "reg_alpha": 20, "reg_lambda": 30, "scale_pos_weight": 48.0},  # [旧池deploy冠军] champion1 全窗+44%（FINAL-DEPLOY同款规格）
    {"pid": "H14", "num_leaves": 127, "min_data_in_leaf": 2000, "feature_fraction": 0.7, "bagging_fraction": 0.75, "reg_alpha": 25, "reg_lambda": 40, "scale_pos_weight": 48.0, "max_depth": 7},  # [旧H8规格] H8_cp400 早期段+79%；与现H8(6/12/70)对照=正则/spw轴
    # [PRUNE 2026-09-28] H4 删除：best_med +0.0058 仅单轮、mean_med 全负；低min_leaf端(800)无赢面
    # {"pid": "H4", "num_leaves": 127, "min_data_in_leaf": 800,  "feature_fraction": 0.65, "bagging_fraction": 0.6, "reg_alpha": 12, "reg_lambda": 12,  "scale_pos_weight": 48.0},  # 中位多样性：低min_leaf端 800/0.65/12/12
    # [PRUNE 2026-09-28] H5 删除：仅+I1轮单点+0.0064、其余轮≤+0.0022且多负；高min_leaf端(2700)无赢面
    # {"pid": "H5", "num_leaves": 127, "min_data_in_leaf": 2700, "feature_fraction": 0.5, "bagging_fraction": 0.5, "reg_alpha": 5, "reg_lambda": 5,  "scale_pos_weight": 40.0},   # 高min_leaf端 2700/0.5/5/5（极端保守端探底）
]
LGB_SEED = 42

RESULTS_ROOT = _HERE / "results"
QUICK_DIR = RESULTS_ROOT / f"stage15hp_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FULL_DIR = RESULTS_ROOT / f"stage15hp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_main(args=None):
    spec = importlib.util.spec_from_file_location("mainmod", str(MAIN_SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CS_LABEL_FAMILY = "multihorizon_35"  # [v74 2026-09-05] r5tol两轮quick未胜出(新99=0.0060<旧99=0.0075)，回滚；r5tol分支保留parked
    # [ABLATION 结论 2026-09-21] hardneg=off：消融（H9/H12×3折同种子同快照）证实新 50
    # 因子池下移除后 mean/median +65~164%、win +2~5pp，q10 劣化仍在 GATE 内 → 维持关闭。
    # 主文件 CS_HARDNEG_ENABLE 未动（仍 True），此处进程内覆盖；要恢复 hardneg 删本行。
    # [2026-09-28 resid-addone系列] hardneg 默认关闭：OFF下测因子本身的峰/谷profile；
    #  ON会先压峰再测，测不出"保峰"。部署验证时再切ON对照。
    mod.CS_HARDNEG_ENABLE = False
    print(f"[ABLATION] CS_HARDNEG_ENABLE={mod.CS_HARDNEG_ENABLE} "
          f"(CS_TAIL_WEIGHT_ENABLE={mod.CS_TAIL_WEIGHT_ENABLE} 保持不变)", flush=True)
    # [FEAT-OVERRIDE 2026-09-24] 特征集覆盖（与 deploy 同机制）：因子 A/B 用（drop-one 等）。
    # 同时覆盖 _get_model_factors 与 _get_curated_model_factors_v12（两者同源，缺一列清单不齐）。
    # 覆盖后的 avail 进入 data_fp.features_sha1 → 与常规 run 天然隔离复用。
    if args is not None and getattr(args, "features_file", None):
        fp = Path(args.features_file)
        if not fp.exists():
            raise FileNotFoundError(f"--features_file 不存在: {fp}")
        feats = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(feats, dict):
            feats = feats.get("features") or feats.get("factors") or []
        feats = [str(x) for x in feats]
        if not feats:
            raise RuntimeError(f"--features_file 为空: {fp}")
        mod._get_curated_model_factors_v12 = lambda: list(feats)
        mod._get_model_factors = lambda *a, **k: list(feats)
        print(f"[FEAT] 特征集覆盖: {len(feats)} 个 (来自 {fp})", flush=True)
    return mod


def _new_logger(name: str, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def build_forward_folds(unique_days, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
                        embargo_days=EMBARGO_TRADE_DAYS, cv_end=CV_END, logger=None):
    """严格 forward folds：train 固定长度窗口，test 紧随（中间隔 embargo 个交易日）。"""
    uniq = np.asarray(unique_days)
    n = len(uniq)
    cv_end_ts = pd.Timestamp(cv_end)
    train_len = int(round(train_months * 21))
    test_len = int(round(test_months * 21))
    folds = []
    start_idx = 0
    fid = 1
    while True:
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = test_start_idx + test_len
        if test_end_idx > n:
            break
        test_end_ts = pd.Timestamp(uniq[min(test_end_idx, n - 1)])
        if test_end_ts > cv_end_ts:
            break
        folds.append(dict(
            fold_id=fid,
            train_start=str(pd.Timestamp(uniq[start_idx]).date()),
            train_end=str(pd.Timestamp(uniq[train_end_idx - 1]).date()),
            test_start=str(pd.Timestamp(uniq[test_start_idx]).date()),
            test_end=str(pd.Timestamp(uniq[test_end_idx - 1]).date()),
        ))
        fid += 1
        start_idx += test_len
    if logger:
        logger.info(f"[CV] folds: {len(folds)} (train={train_months}M test={test_months}M embargo={embargo_days}d)")
        for f_ in folds:
            logger.info(f"  Fold{f_['fold_id']}: train {f_['train_start']}~{f_['train_end']} | test {f_['test_start']}~{f_['test_end']}")
    return folds


def _resolve_data_end(logger) -> str:
    """加载止点自动探测: min(因子库最大日, 本地交易日历最大日, 今天)，防硬编码过期静默截窗。"""
    import pyarrow.parquet as _pq
    _here = Path(__file__).resolve().parent
    if DATA_LOAD_END:
        logger.info(f"[DATA] 加载止点=配置值 {DATA_LOAD_END}")
        return DATA_LOAD_END
    cands = []
    try:
        _t = _pq.read_table(str(_here / "factor_lab_data" / "factors.parquet"), columns=["date"])
        cands.append(("factors.parquet", pd.to_datetime(_t.column("date").to_pandas()).max()))
    except Exception as _e:
        logger.warning(f"[DATA] 探测因子库日期失败: {_e}")
    try:
        _cal = pd.read_parquet(str(_here.parent / "local_data" / "calendar" / "trading_calendar.parquet"),
                               columns=["date"])
        cands.append(("trading_calendar", pd.to_datetime(_cal["date"]).max()))
    except Exception as _e:
        logger.warning(f"[DATA] 探测交易日历失败: {_e}")
    cands.append(("today", pd.Timestamp(datetime.now().date())))
    _end = min(c[1] for c in cands)
    logger.info("[DATA] 加载止点自动探测: %s -> 采用 %s",
                ", ".join(f"{k}={pd.Timestamp(v).date()}" for k, v in cands), _end.date())
    return str(_end.date())


def _compute_hot_mask(factor_data, logger):
    """条件样本标记（hot5_q80：ret5当日截面≥80% 且 turn_hist≥60%），对齐 factor_data 行序。

    与条件IC同口径（复用 multi_horizon 的缓存解析与滚动计算）：pct_chg/turnover_rate
    取自 stock_features 缓存，逐个股滚动后按 (date, stock_code) 左连接回训练行；
    未命中 cache 的行视为非 hot（False）。调用方仅在任一 hot 权重 != 1.0 时执行。
    """
    from multi_horizon_factor_test import _resolve_stock_features_cache, _attach_condition_raw
    t0 = time.time()
    sf = pd.read_parquet(_resolve_stock_features_cache(),
                         columns=["stock_code", "date", "pct_chg", "turnover_rate"])
    sf["date"] = pd.to_datetime(sf["date"])
    sf = _attach_condition_raw(sf)
    f = pd.DataFrame({
        "date": pd.to_datetime(factor_data["date"]).dt.normalize().to_numpy(),
        "stock_code": factor_data["stock_code"].astype(str).to_numpy(),
        "_row": np.arange(len(factor_data)),
    })
    m = f.merge(sf[["date", "stock_code", "_cond_ret5", "_cond_turn_hist"]],
                on=["date", "stock_code"], how="left")
    m = m.sort_values("_row").reset_index(drop=True)
    m["_r5cs"] = m.groupby("date")["_cond_ret5"].rank(pct=True)
    hot = ((m["_r5cs"] >= 0.8) & (m["_cond_turn_hist"] >= 0.6)).fillna(False).to_numpy(dtype=bool)
    logger.info(f"[HOT-W] hot掩码: 命中cache={(m['_cond_ret5'].notna().mean()*100):.1f}% "
                f"hot占比={(hot.mean()*100):.1f}% 耗时={time.time()-t0:.0f}s")
    return hot


def load_and_prepare(main, args, logger):
    """加载因子数据 + 标签 + 特征矩阵（两阶段共用，只加载一次）"""
    data_path = Path(args.factors_path) if args.factors_path else (Path(__file__).resolve().parent / "factor_lab_data" / "factors.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"因子数据不存在: {data_path}  请指定--factors_path")
    main.FACTOR_LAB_STORE_PATH = Path(data_path)
    logger.info(f"[DATA] 源: {data_path}")
    _data_end = _resolve_data_end(logger)
    factor_data = main._load_factor_cache_data(include_label_generation_features=True, start_date=DATA_START, end_date=_data_end)
    if factor_data is None or factor_data.empty:
        raise RuntimeError("因子数据加载失败")
    logger.info(f"[DATA] 加载 {len(factor_data):,}行 时间 {factor_data['date'].min()}~{factor_data['date'].max()}")
    if 'stock_code' in factor_data.columns:
        bj = factor_data['stock_code'].astype(str).str.endswith(('.BJ', '.bj'), na=False)
        if int(bj.sum()) > 0:
            factor_data = factor_data.loc[~bj].copy()
    try:
        vmask, _, _ = main._build_valid_cs_mask(factor_data, logger)
        factor_data["valid_cs_mask"] = vmask.astype(bool)
    except Exception as e:
        factor_data["valid_cs_mask"] = True
    factor_data = main._generate_training_labels_for_all_data(factor_data)
    factor_data = factor_data.sort_values(["date", "stock_code"]).reset_index(drop=True)

    feature_columns = main._get_ml_feature_columns_simple(factor_data)
    X, avail = main._prepare_feature_data(factor_data, feature_columns, data_type="全量特征")
    if X is None:
        raise RuntimeError("特征矩阵构建失败")
    y = factor_data["binary_extreme_label"].values.astype(np.float32)
    y_raw = factor_data["future_ret"].values.astype(np.float64)
    dates = pd.to_datetime(factor_data["date"]).dt.normalize().to_numpy()
    codes = factor_data["stock_code"].astype(str).to_numpy()
    trap = factor_data["trap_stop_sample"].values.astype(bool) if "trap_stop_sample" in factor_data.columns else np.zeros(len(factor_data), dtype=bool)
    # [FIX 2026-09-21] shape_pct 改从 store 原始列计算：v1 CURATED 已不含 5 个形态
    # 因子（hardneg 权重层输入），且 _load_factor_cache_data 会按 CURATED 过滤列，
    # factor_data 里根本没有这些列。因此直接从 store parquet 直读缺失的形态列，
    # 逐日分位秩不受缩放影响，与认证配置同口径（max_rally_20d_raw 已不在 store，
    # 与改版前 _compute_shape_pct 的"缺1个继续算"行为一致）。
    _shape_cols = list(main._SHAPE_FEATURES)
    _shape_missing = [c for c in _shape_cols if c not in factor_data.columns]
    if _shape_missing:
        try:
            import pyarrow.parquet as _pq
            _store_names = set(_pq.read_schema(data_path).names)
            _attach = [c for c in _shape_missing if c in _store_names]
        except Exception as _e:
            logger.warning(f"[WEIGHT-SYNC] 读取 store schema 失败: {_e}")
            _attach = []
        if _attach:
            _ex = pd.read_parquet(data_path, columns=["stock_code", "date"] + _attach)
            _ex["date"] = pd.to_datetime(_ex["date"])
            factor_data = factor_data.merge(_ex, on=["stock_code", "date"], how="left")
            logger.info(f"[WEIGHT-SYNC] 形态因子被加载器过滤，已从 store 直读: {_attach}")
    _shape_present = [c for c in _shape_cols if c in factor_data.columns]
    if not _shape_present:
        raise RuntimeError(f"[WEIGHT-SYNC] store 缺全部形态因子 {_shape_cols}，无法计算 shape_pct")
    if len(_shape_present) < len(_shape_cols):
        _missing_shape = sorted(set(_shape_cols) - set(_shape_present))
        if getattr(main, "CS_HARDNEG_ENABLE", False):
            logger.warning(f"[WEIGHT-SYNC] store 缺形态因子 {_missing_shape}，"
                           f"shape_pct 用现有 {len(_shape_present)}/{len(_shape_cols)} 个（与认证配置有偏差）")
        else:
            logger.info(f"[WEIGHT-SYNC] store 缺形态因子 {_missing_shape}；CS_HARDNEG_ENABLE=False，"
                        f"shape_pct 不参与权重（仅满足 _build_binary_negative_weights 非空签名），无实际影响")
    _shape_df = factor_data[_shape_present].apply(pd.to_numeric, errors="coerce").copy()
    _shape_df["__d"] = pd.to_datetime(factor_data["date"]).dt.normalize().to_numpy()
    shape_pct = _shape_df.groupby("__d").rank(pct=True).mean(axis=1).to_numpy(dtype=float)
    # valid_cs_mask 已含 ST+新股+停牌10日隔离（主脚本 _build_valid_cs_mask 统一生成）
    if "valid_cs_mask" in factor_data.columns:
        valid_mask = factor_data["valid_cs_mask"].values.astype(bool)
    else:
        valid_mask = np.ones(len(factor_data), dtype=bool)
    logger.info(f"[FEAT] X={X.shape} 特征={len(avail)} valid_cs_mask有效率={(valid_mask.mean()*100):.1f}%")
    # [REUSE 2026-09-22] 数据/代码指纹：full 复用 quick 落盘行前逐项比对，防跨因子库/跨池/跨代码错配。
    data_fp = None
    try:
        _st = data_path.stat()
        data_fp = {
            "factors_path": str(data_path.resolve()),
            "factors_mtime": int(_st.st_mtime),
            "factors_size": int(_st.st_size),
            "n_rows": int(len(factor_data)),
            "n_features": int(len(avail)),
            "features_sha1": hashlib.sha1("|".join(sorted(map(str, avail))).encode("utf-8")).hexdigest(),
            "train_months": int(args.train_months),
            "cv_end": str(getattr(args, "cv_end", CV_END)),
            "lgb_seed": int(LGB_SEED),
            "main_src_mtime": int(MAIN_SRC.stat().st_mtime) if MAIN_SRC.exists() else 0,
            "label_family": str(getattr(main, "CS_LABEL_FAMILY", "")),
            "hardneg": bool(getattr(main, "CS_HARDNEG_ENABLE", False)),
            "hardneg_w": float(getattr(main, "CS_HARDNEG_WEIGHT", 5.0)),
            "tail_weight": bool(getattr(main, "CS_TAIL_WEIGHT_ENABLE", False)),
            "hot_w_neg": float(getattr(args, "hot_w_neg", 1.0) or 1.0),
            "hot_w_pos": float(getattr(args, "hot_w_pos", 1.0) or 1.0),
        }
        logger.info(f"[REUSE] data_fp: features_sha1={data_fp['features_sha1'][:12]} "
                    f"n_features={len(avail)} n_rows={len(factor_data):,} mtime={data_fp['factors_mtime']}")
    except Exception as _e:
        logger.warning(f"[REUSE] data_fp 生成失败（本轮不复用 quick）: {_e}")
    return dict(X=X, y=y, y_raw=y_raw, dates=dates, codes=codes, trap=trap,
                shape_pct=shape_pct, avail=avail, data=factor_data, valid_mask=valid_mask,
                data_fp=data_fp,
                hot_mask=_compute_hot_mask(factor_data, logger)
                if (float(getattr(args, "hot_w_neg", 1.0) or 1.0) != 1.0
                    or float(getattr(args, "hot_w_pos", 1.0) or 1.0) != 1.0) else None,
                hot_w_neg=float(getattr(args, "hot_w_neg", 1.0) or 1.0),
                hot_w_pos=float(getattr(args, "hot_w_pos", 1.0) or 1.0))


def build_lgb_params(hp_cfg, num_threads):
    params = {
        "objective": "binary", "metric": "None", "boosting_type": "gbdt",
        "learning_rate": 0.02, "max_depth": hp_cfg.get("max_depth", -1), "bagging_freq": 1,
        "verbosity": -1, "random_state": LGB_SEED,
        "bagging_seed": LGB_SEED, "feature_fraction_seed": LGB_SEED,
        "data_random_seed": LGB_SEED, "drop_seed": LGB_SEED,
        "deterministic": True, "force_col_wise": True,
        "num_leaves": hp_cfg["num_leaves"], "min_data_in_leaf": hp_cfg["min_data_in_leaf"],
        "feature_fraction": hp_cfg["feature_fraction"], "bagging_fraction": hp_cfg["bagging_fraction"],
        "reg_alpha": hp_cfg["reg_alpha"], "reg_lambda": hp_cfg["reg_lambda"],
        "scale_pos_weight": hp_cfg["scale_pos_weight"],
    }
    # num_threads 与 n_jobs 是 LightGBM 同义键，二者同时出现时依赖 dict 顺序易混淆 → 只设其一。
    if num_threads is not None:
        params["num_threads"] = int(num_threads)
    else:
        params["n_jobs"] = 1
    return params


def train_and_diag(main, data_bundle, fold, hp_cfg, max_rounds, checkpoints, num_threads, logger):
    """在单个折上训练并输出 rank/stock/importance 明细。返回三个 list。"""
    X, y, y_raw, dates, codes, trap, shape_pct = (
        data_bundle["X"], data_bundle["y"], data_bundle["y_raw"], data_bundle["dates"],
        data_bundle["codes"], data_bundle["trap"], data_bundle["shape_pct"])
    valid_mask = data_bundle.get("valid_mask", np.ones(len(y), dtype=bool))
    dates_s = pd.Series(dates)
    tr_mask = np.asarray((dates_s >= pd.Timestamp(fold["train_start"])) & (dates_s <= pd.Timestamp(fold["train_end"])))
    te_mask = np.asarray((dates_s >= pd.Timestamp(fold["test_start"])) & (dates_s <= pd.Timestamp(fold["test_end"])))
    tr_ok = tr_mask & np.isfinite(y) & np.isfinite(y_raw) & valid_mask  # [FIX 2026-09-07] 训练行同步要求y_raw有限(与te_ok同口径)；此前训练吃不可交易幻影标签而测试干净，造成train/test不对称
    te_ok = te_mask & np.isfinite(y) & np.isfinite(y_raw) & valid_mask
    if int(tr_ok.sum()) < 50000 or int(te_ok.sum()) < 10000:
        logger.warning(f"[Fold{fold['fold_id']}] 样本不足跳过")
        return [], [], []
    X_tr = X[tr_ok]; y_tr = y[tr_ok]; y_raw_tr = y_raw[tr_ok]
    dates_tr = dates[tr_ok]; trap_tr = trap[tr_ok]
    shape_tr = shape_pct[tr_ok] if shape_pct is not None else None
    w_tr = main._build_binary_negative_weights(y_tr, y_raw_tr, trap_tr, shape_tr)
    w_tr = main._apply_regime_shrinkage(w_tr, y_tr, dates_tr).astype(np.float32)
    # [HOT-WEIGHT 2026-09-22] 条件样本加权：hot 行按标签分乘（默认1.0/1.0=关闭）。
    # data_fp 已记录 hot_w_neg/hot_w_pos，quick/full 复用校验会自动隔离不同权重配置。
    # [A1] hot负类×K、hot正类×1：只教大亏模式，不再给已被tail/spw加码的hot赢家叠加。
    _hot_neg = float(data_bundle.get("hot_w_neg", 1.0) or 1.0)
    _hot_pos = float(data_bundle.get("hot_w_pos", 1.0) or 1.0)
    if _hot_neg != 1.0 or _hot_pos != 1.0:
        _hm = data_bundle.get("hot_mask")
        if _hm is not None and len(_hm) == len(y):
            _hot_tr = np.asarray(_hm, dtype=bool)[tr_ok]
            _is_pos = np.asarray(y_tr > 0.5, dtype=bool)
            _mult_arr = np.where(_hot_tr & ~_is_pos, _hot_neg,
                                 np.where(_hot_tr & _is_pos, _hot_pos, 1.0))
            w_tr = (w_tr * _mult_arr).astype(np.float32)
            logger.info(f"[HOT-W] Fold{fold['fold_id']}: hot行占比={(_hot_tr.mean()*100):.1f}% "
                        f"neg×{_hot_neg}/pos×{_hot_pos}")
        else:
            logger.warning("[HOT-W] hot_mask缺失/长度不符，跳过加权")

    X_te = X[te_ok]; y_raw_te = y_raw[te_ok]; dates_te = dates[te_ok]
    codes_te = codes[te_ok]; y_te = y[te_ok]

    import lightgbm as lgb
    params = build_lgb_params(hp_cfg, num_threads)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, params={"feature_pre_filter": False})
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=max_rounds, callbacks=[lgb.log_evaluation(period=0)])
    logger.info(f"[{hp_cfg['pid']}/Fold{fold['fold_id']}] 训练完成 {time.time()-t0:.0f}s")
    dtr._handle = None

    rank_rows, stock_rows, imp_rows = [], [], []
    for n_cp in checkpoints:
        pred = model.predict(X_te, num_iteration=n_cp)
        df = pd.DataFrame({"date": dates_te, "pred": pred, "ret": y_raw_te, "code": codes_te, "label": y_te})
        for d, g in df.groupby("date", sort=False):
            g = g.sort_values("pred", ascending=False).reset_index(drop=True)
            if len(g) < 5:
                continue
            for r in range(1, 6):
                rk_ret = float(g["ret"].iloc[r - 1]) if pd.notna(g["ret"].iloc[r - 1]) else float("nan")
                rank_rows.append({
                    "fold_id": fold["fold_id"], "checkpoint": n_cp,
                    "date": str(pd.Timestamp(d).date()), "rank": r, "return": rk_ret,
                })
            scores = g["pred"].iloc[:5].to_numpy(dtype=float)
            gap_12 = float(scores[0] - scores[1]) if len(scores) >= 2 else float("nan")
            gap_15 = float(scores[0] - scores[1:5].mean()) if len(scores) >= 5 else float("nan")
            gap_disp = float(np.std(scores[:5])) if len(scores) >= 5 else float("nan")
            for rank_idx in range(5):
                stock_rows.append({
                    "fold_id": fold["fold_id"], "checkpoint": n_cp,
                    "date": str(pd.Timestamp(d).date()), "rank": rank_idx + 1,
                    "stock_code": str(g["code"].iloc[rank_idx]),
                    "pred": float(g["pred"].iloc[rank_idx]),
                    "future_ret": float(g["ret"].iloc[rank_idx]),
                    "label": int(g["label"].iloc[rank_idx] > 0.5),
                    "gap_12": gap_12, "gap_15": gap_15, "gap_disp": gap_disp,
                })
    try:
        _feats = model.feature_name()
        _gain = model.feature_importance(importance_type="gain")
        _split = model.feature_importance(importance_type="split")
        # [v4.3-imp 2026-09-17] 实名保存：LightGBM用numpy训练时feature_name()是
        # Column_N泛名，按X列序(=data_bundle["avail"])回填真实因子名，供复盘。
        # 同文件X与avail同源(_prepare_feature_data同一次返回)，错位即抛错不静默。
        _avail = list(data_bundle.get("avail", []))
        assert len(_avail) == len(_feats) == X.shape[1], \
            f"importance映射错位: avail={len(_avail)} feats={len(_feats)} X={X.shape[1]}"
        for _i, (_f, _g, _s) in enumerate(zip(_feats, _gain, _split)):
            _nm = _avail[_i] if 0 <= _i < len(_avail) else _f
            imp_rows.append({"fold_id": fold["fold_id"], "param": hp_cfg["pid"],
                             "feature": _nm, "gain": float(_g), "split": int(_s)})
    except Exception as _e:
        logger.warning(f"[Fold{fold['fold_id']}] 特征重要性获取失败: {_e}")
    del model
    return rank_rows, stock_rows, imp_rows


def summarize_rank(rank_rows, param_col="param"):
    """逐 (param, cp) 汇总 Rank1: mean/median/q10/mdd_min/mdd_mean/win/n_days
    + [CHAMP-RULE 2026-09-24] 折间稳健列: min_fold_mean/pos_folds/champ_score。

    champ_score = 0.5×min_fold_mean + 0.5×mean（maximin 一致性 + 右尾感知，供选冠军）。
    """
    if not rank_rows:
        return pd.DataFrame()
    r = pd.DataFrame(rank_rows)
    r1 = r[r["rank"] == 1]
    rows = []
    for (pid, cp), sub in r1.groupby([param_col, "checkpoint"], sort=False):
        ret = sub["return"].dropna()
        if len(ret) == 0:
            continue
        mdds = []
        for _, g in sub.groupby("fold_id"):
            e = np.cumprod(1 + g.sort_values("date")["return"].values)
            mdds.append(float(((e / np.maximum.accumulate(e)) - 1).min()))
        _fold_means = sub.groupby("fold_id")["return"].mean().dropna()
        _min_fm = float(_fold_means.min()) if len(_fold_means) else float("nan")
        _mean = float(ret.mean())
        rows.append({
            param_col: pid, "cp": int(cp), "n_days": int(len(ret)), "n_folds": int(sub["fold_id"].nunique()),
            "mean": _mean, "median": float(ret.median()),
            "q10": float(ret.quantile(0.10)), "win": float((ret > 0).mean()),
            "mdd_min": float(min(mdds)) if mdds else float("nan"),
            "mdd_mean": float(np.mean(mdds)) if mdds else float("nan"),
            "min_fold_mean": _min_fm,
            "pos_folds": int((_fold_means > 0).sum()),
            "champ_score": (0.5 * _min_fm + 0.5 * _mean) if np.isfinite(_min_fm) else float("nan"),
        })
    return pd.DataFrame(rows)


def summarize_top5(rank_rows, param_col="param"):
    """[TOP5-AUX 2026-09-23] 逐 (param, cp) 统计 Rank1~5 辅助指标（只展示，不参与冠军记分）：

    - r1_mean..r5_mean / r1_med..r5_med：各 rank 逐日收益的均值/中位数
    - mean_mono / med_mono：均值/中位数是否非递增（r1>=r2>=...>=r5；健康模型应为True）
    - top5_mean / top5_med：先按日取 rank1~5 收益均值，再跨日求均值/中位数
    """
    if not rank_rows:
        return pd.DataFrame()
    r = pd.DataFrame(rank_rows)
    rows = []
    for (pid, cp), sub in r.groupby([param_col, "checkpoint"], sort=False):
        rec = {param_col: pid, "cp": int(cp),
               "n_days": int(sub["date"].nunique()), "n_folds": int(sub["fold_id"].nunique())}
        means, meds, ok = [], [], True
        for rk in range(1, 6):
            v = sub.loc[sub["rank"] == rk, "return"].dropna()
            if len(v) == 0:
                ok = False
                means.append(float("nan")); meds.append(float("nan"))
                continue
            means.append(float(v.mean())); meds.append(float(v.median()))
        for rk in range(1, 6):
            rec[f"r{rk}_mean"] = means[rk - 1]
            rec[f"r{rk}_med"] = meds[rk - 1]
        rec["mean_mono"] = bool(ok and all(means[i] >= means[i + 1] for i in range(4)))
        rec["med_mono"] = bool(ok and all(meds[i] >= meds[i + 1] for i in range(4)))
        try:
            piv = sub.pivot_table(index="date", columns="rank", values="return", aggfunc="mean")
            _cols = [c for c in range(1, 6) if c in piv.columns]
            if len(_cols) == 5:
                daily_top5 = piv[_cols].mean(axis=1).dropna()
                rec["top5_mean"] = float(daily_top5.mean()) if len(daily_top5) else float("nan")
                rec["top5_med"] = float(daily_top5.median()) if len(daily_top5) else float("nan")
            else:
                rec["top5_mean"] = float("nan"); rec["top5_med"] = float("nan")
        except Exception:
            rec["top5_mean"] = float("nan"); rec["top5_med"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


CHAMPION_RULES = {
    "min+mean": ("champ_score", "0.5×min_fold_mean + 0.5×mean（默认）"),
    "median": ("median", "rank1 median（历史口径）"),
    "mean": ("mean", "rank1 mean"),
    "min_fold_mean": ("min_fold_mean", "最差折 mean（maximin）"),
    "pos_folds": ("pos_folds", "正收益折数（同分用 champ_score→median 破平）"),
    # [SLOT-SIM 2026-09-25] 日频槽位组合净值口径（真实可实现，非每日满仓复利）：
    #   2槽=与现行 rqalpha 引擎一致；3槽=与 2~3 日持仓匹配。成本按引擎口径 0.16% 往返。
    "slot2_sharpe": ("slot2_sharpe", "2槽组合模拟 Sharpe（成本0.16%往返）"),
    "slot3_sharpe": ("slot3_sharpe", "3槽组合模拟 Sharpe（成本0.16%往返）"),
    "min+mean+slot2": ("combo_minmean_slot2", "champ_score 与 slot2_sharpe 的 rank 百分位等权混合"),
}

# [SLOT-SIM 2026-09-25] 槽位模拟单边成本：与 rqalpha 引擎 TRADING_COST_TOTAL=0.0016 往返对齐。
SLOT_SIM_COST = 0.0008


def pick_champion(summary, logger, rule="min+mean"):
    """硬门槛 MDD_min>-60% 且 q10>-10% → 按 rule 排序最高；无过门槛者取全组。

    [CHAMP-RULE 2026-09-24] 默认 min+mean。依据：多切点时序验证（选 F1..Fk → 评 F(k+1)，
    k=2..5，三权重平均）+ 全 6 折复核——median 口径在 ONw5 选出 H4@450（OOS 分钟 -2.7%），
    而 min+mean/mean/min_fold_mean 选出 H8@300（OOS 分钟 +79.9%）；诊断根因=median 忽略右尾
    （复利由大赢家驱动）+ 冠军间差距在 bootstrap 噪声内 + 未用折间一致性。median 保留可选。
    """
    if summary is None or summary.empty:
        raise RuntimeError("quick 阶段无任何结果，无法选冠军")
    ok = summary[(summary["mdd_min"] > GATE_MDD_MAX) & (summary["q10"] > GATE_Q10_MIN)]
    if len(ok) == 0:
        logger.warning("[CHAMPION] 无配置通过硬门槛，回退全组（仍按 rule 排序）")
        ok = summary
    col, desc = CHAMPION_RULES.get(rule, CHAMPION_RULES["min+mean"])
    if col not in ok.columns:
        logger.warning(f"[CHAMPION] rule={rule} 所需列 {col} 缺失，回退 median")
        col, desc = "median", "rank1 median（回退）"
    sort_cols = [col] + [c for c in ("champ_score", "median") if c != col and c in ok.columns]
    best = ok.sort_values(sort_cols, ascending=False).iloc[0]
    logger.info(f"[CHAMPION-RULE] rule={rule}（{desc}），排序键={sort_cols}")
    return best


def _attach_slot_metrics(summary, stock_rows, logger, slots=(2, 3), rank=1):
    """[SLOT-SIM 2026-09-25] 给 quick 汇总附加日频槽位组合净值指标（真实口径）。

    - 每配置独立模拟：rank1 picks → 信号 t 次日 open 买入、3 日持有、0.7/0.3 收盘混合卖出，
      N 槽轮换、无空槽入场跳过、单边成本 SLOT_SIM_COST（引擎口径 0.16% 往返）。
    - 部署级验证：同窗 6 配置的 2槽排序与 rqalpha 分钟引擎完全一致。
    - 失败/缺价格缓存时不附加（返回原表），不影响主流程。
    """
    if summary is None or summary.empty or not stock_rows:
        return summary
    try:
        import importlib.util
        _p = Path(__file__).resolve().parent / "stage15_slot_sim.py"
        _spec = importlib.util.spec_from_file_location("slot_sim", str(_p))
        sim = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sim)
        cache = sim._find_prices()
        by_code, cal = sim.load_prices(cache)
        cal_idx = {d: i for i, d in enumerate(cal)}
        det = pd.DataFrame(stock_rows)
        rows = []
        for (pid, cp), g in det.groupby(["param", "checkpoint"], sort=False):
            rec = {"param": pid, "cp": int(cp)}
            for sl in slots:
                r = sim.simulate_group(g, by_code, cal, cal_idx, int(sl), rank, SLOT_SIM_COST)
                if r:
                    for k in ("nav_end", "ann", "sharpe", "maxdd", "trades", "skipped"):
                        rec[f"slot{sl}_{k}"] = r[k]
            rows.append(rec)
        sm = pd.DataFrame(rows)
        out = summary.merge(sm, on=["param", "cp"], how="left")
        if "champ_score" in out.columns and "slot2_sharpe" in out.columns:
            out["combo_minmean_slot2"] = (0.5 * out["champ_score"].rank(pct=True)
                                          + 0.5 * out["slot2_sharpe"].rank(pct=True))
        logger.info(f"[SLOT-SIM] 已附加槽位指标 slots={list(slots)} 成本(单边)={SLOT_SIM_COST} "
                    f"覆盖={len(sm)}/{len(summary)}（缓存={cache.name}）")
        return out
    except Exception as e:
        logger.warning(f"[SLOT-SIM] 槽位指标计算失败，跳过（不影响原流程）: {e}")
        return summary


def _atomic_csv(df, path, logger):
    """原子写CSV（tmp+os.replace），防kill残留半文件。"""
    _tmp = Path(str(path) + ".tmp")
    df.to_csv(_tmp, index=False, encoding="utf-8-sig")
    os.replace(_tmp, path)


def _write_progress(path, done, total, t_start, extra=None):
    """心跳文件：done/total/已耗时/ETA。给watch tail和队列父进程看进度用。"""
    el = time.time() - t_start
    rate = len(done) / max(el, 1)
    eta = int((total - len(done)) / rate) if rate > 0 else -1
    payload = {"done": sorted(done), "total": int(total),
               "elapsed_s": int(el), "eta_s": eta,
               "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        payload.update(extra)
    _tmp = Path(str(path) + ".tmp")
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(_tmp, path)
    return eta


def _quick_reuse_ok(qdir, pid, data_bundle, logger):
    """校验 quick 目录可否被 full 复用：协议(轮数/cp) + 超参定义 + 数据/代码指纹 三者全等。"""
    cfg_path = qdir / "hp_config.json"
    if not cfg_path.exists():
        return False, None, "缺 hp_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as _e:
        return False, None, f"hp_config.json 解析失败: {_e}"
    if cfg.get("phase") != "quick":
        return False, cfg, "phase != quick"
    if int(cfg.get("rounds") or 0) < FULL_ROUNDS:
        return False, cfg, f"rounds={cfg.get('rounds')} < FULL_ROUNDS={FULL_ROUNDS}（旧协议无法提供高轮 cp）"
    _cps = set(int(c) for c in (cfg.get("checkpoints") or []))
    if not set(int(c) for c in FULL_CHECKPOINTS).issubset(_cps):
        return False, cfg, f"checkpoints={sorted(_cps)} 不含 FULL_CHECKPOINTS"
    _hp_cur = next((h for h in HP_GRID if h["pid"] == pid), None)
    _hp_q = next((h for h in (cfg.get("hp_grid") or []) if h.get("pid") == pid), None)
    if _hp_cur is None or _hp_q is None or dict(_hp_cur) != dict(_hp_q):
        return False, cfg, f"超参 {pid} 定义不一致（quick={_hp_q} / 当前={_hp_cur}）"
    if data_bundle.get("data_fp") is None:
        return False, cfg, "本轮 data_fp 不可用"
    if cfg.get("data_fp") != data_bundle["data_fp"]:
        return False, cfg, "data_fp 不一致（因子库/特征集/标签/权重配置/训练窗口已变）"
    return True, cfg, "ok"


def _load_reuse_rows(qdir, pid, folds, logger):
    """读 quick 落盘三件套，按 pid×折×cp 过滤出可复用行（rank/stock 去 param 对齐 full 表头）。

    仅当某折的 rank 行覆盖全部 FULL_CHECKPOINTS 且 stock/imp 行齐全时才复用该折。
    """
    out = {"rank": [], "stock": [], "imp": [], "folds": []}
    fids = {int(f["fold_id"]) for f in folds}
    cps = set(int(c) for c in FULL_CHECKPOINTS)
    frames = {}
    for _fn, _k in (("hp_rank_daily.csv", "rank"), ("hp_stock_details_top5.csv", "stock"),
                    ("hp_feature_importance.csv", "imp")):
        _fp = qdir / _fn
        if not _fp.exists():
            logger.warning(f"[REUSE] 缺 {_fn}，放弃复用")
            return out
        try:
            _df = pd.read_csv(_fp, encoding="utf-8-sig")
        except Exception as _e:
            logger.warning(f"[REUSE] {_fn} 读取失败: {_e}")
            return out
        if "param" in _df.columns:
            _df = _df[_df["param"].astype(str) == str(pid)]
        frames[_k] = _df
    _rk, _st, _im = frames["rank"], frames["stock"], frames["imp"]
    if not len(_rk) or not len(_st) or not len(_im):
        return out
    _rk = _rk[_rk["fold_id"].isin(fids) & _rk["checkpoint"].isin(cps)]
    _st = _st[_st["fold_id"].isin(fids) & _st["checkpoint"].isin(cps)]
    _im = _im[_im["fold_id"].isin(fids)]
    _fold_rank_cps = {int(f): set(int(c) for c in g["checkpoint"]) for f, g in _rk.groupby("fold_id")}
    _fold_st = set(int(f) for f in _st["fold_id"])
    _fold_im = set(int(f) for f in _im["fold_id"])
    good = sorted(f for f in fids
                  if f in _fold_im and f in _fold_st and _fold_rank_cps.get(f, set()) >= cps)
    if not good:
        return out
    _drop = lambda df: df.drop(columns=[c for c in ("param",) if c in df.columns])
    out["folds"] = good
    out["rank"] = _drop(_rk[_rk["fold_id"].isin(good)]).to_dict("records")
    out["stock"] = _drop(_st[_st["fold_id"].isin(good)]).to_dict("records")
    out["imp"] = _im[_im["fold_id"].isin(good)].to_dict("records")
    return out


def _find_reusable_quick_dir(pid, data_bundle, logger, explicit=None):
    """复用源选择：显式目录只验它；否则按 mtime 倒序扫描 stage15hp_quick_* 取第一个全过校验者。"""
    if explicit:
        _c = Path(explicit)
        if not _c.exists():
            logger.warning(f"[REUSE] 指定 quick 目录不存在: {_c}")
            return None
        ok, _cfg, _why = _quick_reuse_ok(_c, pid, data_bundle, logger)
        logger.info(f"[REUSE] {_c.name}: {'可复用' if ok else '不复用 → ' + _why}")
        return _c if ok else None
    _cands = sorted([p for p in RESULTS_ROOT.glob("stage15hp_quick_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True)
    for _c in _cands:
        ok, _cfg, _why = _quick_reuse_ok(_c, pid, data_bundle, logger)
        if ok:
            logger.info(f"[REUSE] 自动选中 quick 目录 {_c.name}（pid={pid}）")
            return _c
    logger.info(f"[REUSE] 未找到可复用的 quick 目录（pid={pid}），全部折重训")
    return None


def _find_latest_champion_dir(logger):
    """扫描最新含 champion.json 的 quick 目录（full 未显式指定 --quick_dir 时读冠军用）。"""
    _cands = sorted([p for p in RESULTS_ROOT.glob("stage15hp_quick_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True)
    for _c in _cands:
        if (_c / "champion.json").exists():
            return _c
    return None


def run_quick(main, data_bundle, folds, args, logger):
    """阶段A：HP_GRID 全组 × 全部折(默认1~6) × 600轮 × 全7档cp，num_threads=3。"""
    qdir = Path(args.quick_dir_out) if getattr(args, "quick_dir_out", None) else QUICK_DIR
    qdir.mkdir(parents=True, exist_ok=True)
    quick_folds_ids = QUICK_FOLD_IDS
    _qf = getattr(args, "quick_folds", None)
    if _qf:
        try:
            quick_folds_ids = [int(x.strip()) for x in str(_qf).split(",") if x.strip()]
        except ValueError:
            pass
    quick_folds = [f for f in folds if f["fold_id"] in quick_folds_ids]
    if len(quick_folds) < len(quick_folds_ids):
        logger.warning(f"[QUICK] 期望折 {quick_folds_ids}，实际只有 {[f['fold_id'] for f in quick_folds]}（train_months 可能非18）")
    if not quick_folds:
        raise RuntimeError("无可用的代表性折")

    rank_rows, stock_rows, imp_rows = [], [], []
    hp_list = HP_GRID if not getattr(args, "hp_names", None) else [h for h in HP_GRID if h["pid"] in args.hp_names]
    if not hp_list:
        raise RuntimeError("hp_names 过滤后无超参")
    logger.info("=" * 80)
    logger.info(f"[阶段A/QUICK] {len(hp_list)}组超参{[h['pid'] for h in hp_list]} × {len(quick_folds_ids)}折{quick_folds_ids} "
                f"× {QUICK_TRAIN_ROUNDS}轮 × cp{QUICK_ALL_CHECKPOINTS} | num_threads={QUICK_NUM_THREADS}"
                f"（冠军候选仍为 cp{QUICK_CHECKPOINTS}；全量 cp 落盘供 full 复用）")
    logger.info("=" * 80)
    print(f"[QUICK] {len(hp_list)}组超参 × {len(quick_folds)}折 = {len(hp_list)*len(quick_folds)}训练单元", flush=True)
    logger.info(f"[QUICK] {len(hp_list)}组超参{[h['pid'] for h in hp_list]} × {len(quick_folds)}折")
    # [断点续跑 2026-09-17] 预载已落盘行，只保留当前网格+折，跳过已完成(param,fold)。
    # 杀进程只丢"正在训的1个单元"，重启自动续跑（幂等purge保证重算不翻倍）。
    for _fn, _which in (("hp_rank_daily.csv", "rank"), ("hp_stock_details_top5.csv", "stock"),
                        ("hp_feature_importance.csv", "imp")):
        _fp = qdir / _fn
        if _fp.exists():
            try:
                _old = pd.read_csv(_fp, encoding="utf-8-sig")
                if _which == "rank":
                    rank_rows = _old.to_dict("records")
                elif _which == "stock":
                    stock_rows = _old.to_dict("records")
                else:
                    imp_rows = _old.to_dict("records")
                logger.info(f"[RESUME] 载入已落盘{_fn}: {len(_old):,}行")
            except Exception as _e:
                logger.warning(f"[RESUME] {_fn} 读取失败，从头跑: {_e}")
    _keep_pids = {str(h["pid"]) for h in hp_list}
    _keep_fids = {int(f["fold_id"]) for f in quick_folds}

    def _keep(r):
        try:
            return str(r.get("param")) in _keep_pids and int(r.get("fold_id", -1)) in _keep_fids
        except Exception:
            return False

    rank_rows = [r for r in rank_rows if _keep(r)]
    stock_rows = [r for r in stock_rows if _keep(r)]
    imp_rows = [r for r in imp_rows if _keep(r)]
    _done = {(str(r["param"]), int(r["fold_id"])) for r in imp_rows}
    _total = len(hp_list) * len(quick_folds)
    if _done:
        logger.info(f"[RESUME] 已完成 {len(_done)}/{_total} 个(param,fold)，跳过续跑")
    print(f"[QUICK] 已完成{len(_done)}，待跑{_total - len(_done)}", flush=True)
    t_start = time.time()
    for hp in hp_list:
        _hp_t0 = time.time()
        for fold in quick_folds:
            _key = (str(hp["pid"]), int(fold["fold_id"]))
            if _key in _done:
                continue
            rr, sr, ir = train_and_diag(main, data_bundle, fold, hp, QUICK_TRAIN_ROUNDS, QUICK_ALL_CHECKPOINTS,
                                        QUICK_NUM_THREADS, logger)
            for row in rr:
                row["param"] = hp["pid"]
            for row in sr:
                row["param"] = hp["pid"]
            for row in ir:
                row["param"] = hp["pid"]
            _purge = lambda rows: [r for r in rows if not (
                str(r.get("param")) == _key[0] and int(r.get("fold_id", -999)) == _key[1])]
            rank_rows, stock_rows, imp_rows = _purge(rank_rows), _purge(stock_rows), _purge(imp_rows)
            rank_rows.extend(rr); stock_rows.extend(sr); imp_rows.extend(ir)
            _done.add(_key)
        if rank_rows:
            _atomic_csv(pd.DataFrame(rank_rows), qdir / "hp_rank_daily.csv", logger)
        if stock_rows:
            _atomic_csv(pd.DataFrame(stock_rows), qdir / "hp_stock_details_top5.csv", logger)
        if imp_rows:
            _atomic_csv(pd.DataFrame(imp_rows), qdir / "hp_feature_importance.csv", logger)
        _el = time.time() - t_start
        _eta = int((_el / max(len(_done), 1)) * (_total - len(_done))) if len(_done) < _total else 0
        _msg = (f"[QUICK] HP {hp['pid']} done {len(quick_folds)}folds/{time.time()-_hp_t0:.0f}s | "
                f"{len(_done)}/{_total}单元, 已跑{_el/60:.0f}min, ETA~{_eta//60}min")
        logger.info(_msg)
        print(_msg, flush=True)
        _write_progress(qdir / "progress.json", [f"{p}:{f}" for p, f in sorted(_done)],
                        _total, t_start, extra={"phase": "quick", "hp": hp["pid"]})
    logger.info(f"[QUICK] 全部训练完成 耗时 {time.time()-t_start:.0f}s")

    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(qdir / "hp_rank_daily.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] hp_rank_daily.csv: {len(rank_rows):,}行 (param列, 全cp×全rank逐日)")
    if stock_rows:
        pd.DataFrame(stock_rows).to_csv(qdir / "hp_stock_details_top5.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] hp_stock_details_top5.csv: {len(stock_rows):,}行")
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(qdir / "hp_feature_importance.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] hp_feature_importance.csv: {len(imp_rows):,}行")

    # [EXPAND 2026-09-23] 冠军候选 cp = QUICK_CHECKPOINTS（现为全 7 档）。
    _sel_cps = set(int(c) for c in QUICK_CHECKPOINTS)
    _sel_rows = [r for r in rank_rows if int(r["checkpoint"]) in _sel_cps]
    summary = summarize_rank(_sel_rows)
    # [SLOT-SIM 2026-09-25] 附加槽位组合净值指标（真实口径；失败自动跳过）。
    summary = _attach_slot_metrics(summary, stock_rows, logger)
    if len(summary):
        summary.to_csv(qdir / "hp_summary.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] hp_summary.csv: {len(summary):,}行 (param×cp 汇总)")
        _base_cols = ["param", "cp", "n_days", "n_folds", "mean", "median", "q10", "win",
                      "mdd_min", "mdd_mean", "min_fold_mean", "pos_folds", "champ_score"]
        print("\n===== 快速搜索汇总 (param × cp, Rank1) =====", flush=True)
        print(summary[[c for c in _base_cols if c in summary.columns]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
        if "slot2_sharpe" in summary.columns:
            _scols = ["param", "cp", "slot2_nav_end", "slot2_ann", "slot2_sharpe", "slot2_maxdd",
                      "slot2_skipped", "slot3_nav_end", "slot3_ann", "slot3_sharpe", "slot3_maxdd"]
            print("\n===== 槽位组合模拟 (2/3槽, rank1, 成本0.16%往返; 日频真实净值) =====", flush=True)
            print(summary[[c for c in _scols if c in summary.columns]]
                  .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    # [TOP5-AUX 2026-09-23] Rank1~5 单调性 + Top5 均收益：只统计打印辅助判断冠军，不参与记分。
    top5 = summarize_top5(_sel_rows)
    if len(top5):
        top5.to_csv(qdir / "hp_top5_summary.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] hp_top5_summary.csv: {len(top5):,}行 (param×cp Top5辅助)")
        print("\n===== Top5 辅助诊断 (r1~r5 均值/中位数, 单调性, Top5均收益; 不参与冠军记分) =====", flush=True)
        print(top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    _rule = str(getattr(args, "champion_rule", "min+mean") or "min+mean")
    best = pick_champion(summary, logger, rule=_rule)
    # [CHAMP-RULE 2026-09-24] 对照：median 口径冠军（历史规则）一并记录，便于复盘规则差异。
    _best_med = pick_champion(summary, logger, rule="median")
    champ = dict(pid=str(best["param"]), cp=int(best["cp"]), median=float(best["median"]),
                 mean=float(best["mean"]), q10=float(best["q10"]), mdd_min=float(best["mdd_min"]),
                 champion_rule=_rule,
                 min_fold_mean=float(best.get("min_fold_mean", float("nan"))),
                 pos_folds=int(best.get("pos_folds", -1)),
                 champ_score=float(best.get("champ_score", float("nan"))),
                 median_rule_pick={"pid": str(_best_med["param"]), "cp": int(_best_med["cp"]),
                                   "median": float(_best_med["median"])})
    # [SLOT-SIM 2026-09-25] 冠军的槽位组合指标（若已计算）
    for _k in ("slot2_nav_end", "slot2_ann", "slot2_sharpe", "slot2_maxdd", "slot2_skipped",
               "slot3_nav_end", "slot3_ann", "slot3_sharpe", "slot3_maxdd"):
        if _k in best.index and pd.notna(best.get(_k)):
            champ[_k] = float(best[_k])
    # [TOP5-AUX] 冠军的 Top5 辅助指标附在 champion.json（仅参考，不参与选择）
    _brow = top5[(top5["param"].astype(str) == str(best["param"])) & (top5["cp"] == int(best["cp"]))] \
        if len(top5) else pd.DataFrame()
    if len(_brow):
        _b = _brow.iloc[0]
        champ.update({"mean_mono": bool(_b["mean_mono"]), "med_mono": bool(_b["med_mono"]),
                      "top5_mean": float(_b["top5_mean"]), "top5_med": float(_b["top5_med"])})
    cfg = {
        "exp": EXP_TAG, "phase": "quick", "train_months": getattr(args, "train_months", TRAIN_MONTHS),
        "hp_grid": hp_list, "quick_fold_ids": quick_folds_ids, "rounds": QUICK_TRAIN_ROUNDS,
        "checkpoints": QUICK_ALL_CHECKPOINTS, "select_checkpoints": QUICK_CHECKPOINTS,
        "num_threads": QUICK_NUM_THREADS, "data_fp": data_bundle.get("data_fp"),
        "hardneg_w": float(getattr(main, "CS_HARDNEG_WEIGHT", 5.0)),
        "champion_rule": _rule,
        "champion": champ,
    }
    with open(qdir / "hp_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(qdir / "champion.json", "w", encoding="utf-8") as f:
        json.dump(champ, f, ensure_ascii=False, indent=2)
    logger.info(f"[CHAMPION] rule={_rule} → {best['param']} @ cp{int(best['cp'])} "
                f"median={float(best['median']):.4f} mean={float(best['mean']):.4f} "
                f"min_fold_mean={float(best.get('min_fold_mean', float('nan'))):.4f} "
                f"pos_folds={int(best.get('pos_folds', -1))} "
                f"q10={float(best['q10']):.4f} mdd_min={float(best['mdd_min']):.4f}")
    if "slot2_sharpe" in best.index and pd.notna(best.get("slot2_sharpe")):
        logger.info(f"[CHAMPION-SLOT] slot2: nav={float(best['slot2_nav_end']):.4f} "
                    f"ann={float(best['slot2_ann']):+.4f} sharpe={float(best['slot2_sharpe']):+.4f} "
                    f"mdd={float(best['slot2_maxdd']):+.4f} skipped={int(best.get('slot2_skipped', -1))} | "
                    f"slot3: sharpe={float(best.get('slot3_sharpe', float('nan'))):+.4f}")
    if str(_best_med["param"]) != str(best["param"]) or int(_best_med["cp"]) != int(best["cp"]):
        logger.info(f"[CHAMPION-MEDIAN-REF] median口径冠军={_best_med['param']}@cp{int(_best_med['cp'])} "
                    f"median={float(_best_med['median']):.4f}（仅对照，未采用）")
    if len(_brow):
        logger.info(f"[CHAMPION-TOP5] mean_mono={bool(_b['mean_mono'])} med_mono={bool(_b['med_mono'])} "
                    f"top5_mean={float(_b['top5_mean']):.4f} top5_med={float(_b['top5_med']):.4f}（仅参考）")
    logger.info(f"[DONE] 快速搜索目录: {qdir}")
    return champ, qdir


def run_full(main, data_bundle, folds, champ, args, logger, reuse_dir=None):
    """阶段B：冠军超参 × 全部折 × 600轮 × 7cp，num_threads=FULL_NUM_THREADS（跨线程逐bit一致已实测）。"""
    fdir = Path(args.full_dir_out) if getattr(args, "full_dir_out", None) else FULL_DIR
    fdir.mkdir(parents=True, exist_ok=True)
    _full_threads = int(getattr(args, "full_threads", FULL_NUM_THREADS))
    hp = next((h for h in HP_GRID if h["pid"] == champ["pid"]), None)
    if hp is None:
        raise RuntimeError(f"冠军超参 {champ['pid']} 不在 HP_GRID 中")
    logger.info("=" * 80)
    logger.info(f"[阶段B/FULL] 冠军 {hp['pid']} × {len(folds)}折 × {FULL_ROUNDS}轮 × {FULL_CHECKPOINTS} | num_threads={_full_threads}")
    logger.info("=" * 80)

    # [断点续跑 2026-09-17] 预载已落盘行，跳过已完成fold（幂等purge防翻倍）。
    rank_rows, stock_rows, imp_rows = [], [], []
    for _fn, _which in (("rank_daily.csv", "rank"), ("stock_details_top5.csv", "stock"),
                        ("feature_importance.csv", "imp")):
        _fp = fdir / _fn
        if _fp.exists():
            try:
                _old = pd.read_csv(_fp, encoding="utf-8-sig")
                if _which == "rank":
                    rank_rows = _old.to_dict("records")
                elif _which == "stock":
                    stock_rows = _old.to_dict("records")
                else:
                    imp_rows = _old.to_dict("records")
                logger.info(f"[RESUME] 载入已落盘{_fn}: {len(_old):,}行")
            except Exception as _e:
                logger.warning(f"[RESUME] {_fn} 读取失败，从头跑: {_e}")
    _keep_fids = {int(f["fold_id"]) for f in folds}
    rank_rows = [r for r in rank_rows if int(r.get("fold_id", -1)) in _keep_fids]
    stock_rows = [r for r in stock_rows if int(r.get("fold_id", -1)) in _keep_fids]
    imp_rows = [r for r in imp_rows if int(r.get("fold_id", -1)) in _keep_fids]
    _done = {int(r["fold_id"]) for r in imp_rows}
    _total = len(folds)
    if _done:
        logger.info(f"[RESUME] 已完成 {sorted(_done)}，跳过续跑")
    # [REUSE 2026-09-22] 复用 quick 已训练的折：同 data_fp + 同超参定义 + quick 全量 cp 落盘，
    # 公共行实测与 full 逐bit一致 → 免重训。先清同折旧行再灌入（幂等，防半文件残留翻倍）。
    reused_folds = []
    if reuse_dir is not None:
        _reuse = _load_reuse_rows(reuse_dir, hp["pid"], folds, logger)
        _new_reuse = [f for f in _reuse["folds"] if f not in _done]
        if _new_reuse:
            _rs = set(_new_reuse)
            _purge_r = [r for r in rank_rows if int(r.get("fold_id", -999)) not in _rs]
            _purge_s = [r for r in stock_rows if int(r.get("fold_id", -999)) not in _rs]
            _purge_i = [r for r in imp_rows if int(r.get("fold_id", -999)) not in _rs]
            rank_rows, stock_rows, imp_rows = _purge_r, _purge_s, _purge_i
            rank_rows.extend([r for r in _reuse["rank"] if int(r["fold_id"]) in _rs])
            stock_rows.extend([r for r in _reuse["stock"] if int(r["fold_id"]) in _rs])
            imp_rows.extend([r for r in _reuse["imp"] if int(r["fold_id"]) in _rs])
            _done.update(_new_reuse)
            reused_folds = _new_reuse
            logger.info(f"[REUSE] 复用 {reuse_dir.name} 的 Fold{reused_folds}（pid={hp['pid']}，"
                        f"{QUICK_TRAIN_ROUNDS}轮/num_threads={QUICK_NUM_THREADS} 产物，同协议逐bit一致），免重训")
            _atomic_csv(pd.DataFrame(rank_rows), fdir / "rank_daily.csv", logger)
            _atomic_csv(pd.DataFrame(stock_rows), fdir / "stock_details_top5.csv", logger)
            _atomic_csv(pd.DataFrame(imp_rows), fdir / "feature_importance.csv", logger)
        else:
            logger.info(f"[REUSE] {reuse_dir.name} 无可新增复用的折（或无完整落盘行），全部重训")
    _msg0 = f"[FULL] {len(folds)}折，已完成{len(_done)}（其中复用{len(reused_folds)}），待跑{_total - len(_done)}"
    logger.info(_msg0)
    print(_msg0, flush=True)
    t_start = time.time()
    _write_progress(fdir / "progress.json", [str(f) for f in sorted(_done)],
                    _total, t_start, extra={"phase": "full", "pid": hp["pid"],
                                            "reused_folds": reused_folds})
    for fold in folds:
        _fid = int(fold["fold_id"])
        if _fid in _done:
            continue
        _fold_t0 = time.time()
        rr, sr, ir = train_and_diag(main, data_bundle, fold, hp, FULL_ROUNDS, FULL_CHECKPOINTS,
                                    _full_threads, logger)
        _purge = lambda rows: [r for r in rows if int(r.get("fold_id", -999)) != _fid]
        rank_rows, stock_rows, imp_rows = _purge(rank_rows), _purge(stock_rows), _purge(imp_rows)
        rank_rows.extend(rr); stock_rows.extend(sr); imp_rows.extend(ir)
        _done.add(_fid)
        if rank_rows:
            _atomic_csv(pd.DataFrame(rank_rows), fdir / "rank_daily.csv", logger)
        if stock_rows:
            _atomic_csv(pd.DataFrame(stock_rows), fdir / "stock_details_top5.csv", logger)
        if imp_rows:
            _atomic_csv(pd.DataFrame(imp_rows), fdir / "feature_importance.csv", logger)
        _el = time.time() - t_start
        _eta = int((_el / max(len(_done), 1)) * (_total - len(_done))) if len(_done) < _total else 0
        _msg = (f"[FULL] Fold{_fid} done/{time.time()-_fold_t0:.0f}s本折 | "
                f"{len(_done)}/{_total}折, 已跑{_el/60:.0f}min, ETA~{_eta//60}min")
        logger.info(_msg)
        print(_msg, flush=True)
        _write_progress(fdir / "progress.json", [str(f) for f in sorted(_done)],
                        _total, t_start, extra={"phase": "full", "pid": hp["pid"]})
    logger.info(f"[FULL] 全部训练完成 耗时 {time.time()-t_start:.0f}s")

    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(fdir / "rank_daily.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] rank_daily.csv: {len(rank_rows):,}行")
    if stock_rows:
        pd.DataFrame(stock_rows).to_csv(fdir / "stock_details_top5.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] stock_details_top5.csv: {len(stock_rows):,}行")
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(fdir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        logger.info(f"[SAVE] feature_importance.csv: {len(imp_rows):,}行")

    # [PRINT 2026-09-22] full 也打印与 quick 同格式的 cp 汇总表（Rank1 跨全折），并落盘 summary.csv
    if rank_rows:
        _summary = summarize_rank([{**r, "param": hp["pid"]} for r in rank_rows])
        if len(_summary):
            _summary.to_csv(fdir / "summary.csv", index=False, encoding="utf-8-sig")
            logger.info(f"[SAVE] summary.csv: {len(_summary):,}行 (cp 汇总)")
            print(f"\n===== 全量验证汇总 ({hp['pid']} × cp, Rank1, {len(folds)}折) =====", flush=True)
            print(_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    # [TOP5-AUX 2026-09-23] full 同样打印/落盘 Top5 辅助诊断（只展示，不参与选择）
    if rank_rows:
        _top5 = summarize_top5([{**r, "param": hp["pid"]} for r in rank_rows])
        if len(_top5):
            _top5.to_csv(fdir / "summary_top5.csv", index=False, encoding="utf-8-sig")
            logger.info(f"[SAVE] summary_top5.csv: {len(_top5):,}行")
            print(f"\n===== 全量验证 Top5 辅助诊断 ({hp['pid']} × cp; 不参与记分) =====", flush=True)
            print(_top5.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    cfg = {
        "exp": EXP_TAG, "phase": "full", "train_months": getattr(args, "train_months", TRAIN_MONTHS),
        "champion": champ, "hp": hp,
        "folds": folds, "checkpoints": FULL_CHECKPOINTS, "max_rounds": FULL_ROUNDS,
        "num_threads": _full_threads,
        "reused_from": str(reuse_dir) if reused_folds else None,
        "reused_folds": reused_folds,
        "data_fp": data_bundle.get("data_fp"),
    }
    with open(fdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger.info(f"[DONE] 全量验证目录: {fdir}")
    return fdir


def parse_args():
    p = argparse.ArgumentParser(description="Stage1.5 两阶段公平超参搜索")
    p.add_argument("--mode", choices=["all", "quick", "full"], default="all",
                   help="all=quick+full / quick=只搜索 / full=只全量验证")
    p.add_argument("--factors_path", default=None, help="因子parquet路径")
    p.add_argument("--cv_end", default=CV_END,
                   help="CV folds upper bound (default %s; use 2025-12-31 for strict V6 A/B)" % CV_END)
    p.add_argument("--train_months", type=int, default=TRAIN_MONTHS, help="训练窗口月数(默认18)")
    p.add_argument("--quick_dir", default=None, help="full 模式读取冠军/复用来源的 quick 目录(含 champion.json 与 hp_*.csv)")
    p.add_argument("--force_pid", default=None, help="full 模式强制指定超参pid(跳过champion.json；仍会自动寻找可复用quick目录)")
    p.add_argument("--no_reuse", action="store_true", help="full 模式禁用 quick 折复用，全部折重训")
    p.add_argument("--full_threads", type=int, default=FULL_NUM_THREADS,
                   help=f"full 训练线程数(默认{FULL_NUM_THREADS}，与 quick 同速；deterministic=True 跨线程数逐bit一致已实测)")
    p.add_argument("--hot_w_neg", type=float, default=1.0,
                   help="hot负类(y=0)权重倍数(默认1.0=关闭；A1实验用2.0，只教大亏模式，已记入data_fp防复用错配)")
    p.add_argument("--hot_w_pos", type=float, default=1.0,
                   help="hot正类(y=1)权重倍数(默认1.0=关闭；与hot_w_neg配合，已记入data_fp)")
    p.add_argument("--hardneg_on", action="store_true",
                   help="打开hardneg形态加权(默认关闭；开时用max_rally_20d代替未生成的max_rally_20d_raw补齐5/5，已记入data_fp)")
    p.add_argument("--hardneg_w", type=float, default=None,
                   help="hardneg权重覆盖(默认None=主文件CS_HARDNEG_WEIGHT=5.0；需与--hardneg_on同用；已记入data_fp防复用错配)")
    p.add_argument("--hp_names", default=None, help="quick 模式限定超参子集, 逗号分隔(如 P06,H1)")
    p.add_argument("--features_file", default=None,
                   help="特征集覆盖JSON(list或{features:[...]})；因子A/B用（同源覆盖两个特征函数；features_sha1 入 data_fp 防复用错配）")
    p.add_argument("--quick_dir_out", default=None, help="quick 输出目录覆盖")
    p.add_argument("--quick_folds", default=None, help="quick折子集，逗号分隔(默认1,2,3,4,5,6 全折)")
    p.add_argument("--champion_rule", default="min+mean", choices=list(CHAMPION_RULES.keys()),
                   help="quick 冠军选择规则（默认 min+mean=0.5×min_fold_mean+0.5×mean；"
                        "median=历史口径；slot2_sharpe/slot3_sharpe=日频槽位组合净值口径；"
                        "min+mean+slot2=百分位混合；依据见 pick_champion/_attach_slot_metrics docstring）")
    p.add_argument("--full_dir_out", default=None, help="full 输出目录覆盖")
    return p.parse_args()


def main_flow(args):
    logger = _new_logger("hpsearch", RESULTS_ROOT / f"hpsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"[BOOT] import 主脚本 ...")
    main = _load_main(args)
    if getattr(args, "hardneg_on", False) or bool(getattr(main, "CS_HARDNEG_ENABLE", False)):
        # [E2 2026-09-22] hardneg ON 验证：max_rally_20d_raw 未生成入库，
        # 用 max_rally_20d 代替（risk_control.py:150-151 证实两者同函数同值，替代精确）。
        # [2026-09-28] 默认已开：or 后的条件保证替代逻辑跟随有效开关（而不仅是 --hardneg_on 旗标），
        # 否则默认开启会走到 4/5 形态分支，与旗标行为不一致。
        main.CS_HARDNEG_ENABLE = True
        _sh = list(main._SHAPE_FEATURES)
        if "max_rally_20d_raw" in _sh:
            _sh[_sh.index("max_rally_20d_raw")] = "max_rally_20d"
            main._SHAPE_FEATURES = _sh
        logger.info("[FLAGS] CS_HARDNEG_ENABLE=True（--hardneg_on；shape用max_rally_20d代替max_rally_20d_raw，5/5齐全）")
    if getattr(args, "hardneg_w", None) is not None:
        # [HARDNEG-W 2026-09-24] 权重扫描：仅进程内覆盖主文件常量（默认5.0），已记入 data_fp 防复用错配。
        if not getattr(args, "hardneg_on", False):
            logger.warning(f"[FLAGS] --hardneg_w={args.hardneg_w} 但未开 --hardneg_on，权重不生效")
        else:
            main.CS_HARDNEG_WEIGHT = float(args.hardneg_w)
            logger.info(f"[FLAGS] hardneg权重覆盖: CS_HARDNEG_WEIGHT={main.CS_HARDNEG_WEIGHT:g}")
    logger.info(f"[FLAGS] CS_HARDNEG_ENABLE={getattr(main, 'CS_HARDNEG_ENABLE', None)} "
                f"CS_HARDNEG_WEIGHT={getattr(main, 'CS_HARDNEG_WEIGHT', None)} "
                f"CS_TAIL_WEIGHT_ENABLE={getattr(main, 'CS_TAIL_WEIGHT_ENABLE', None)} "
                f"CS_LABEL_FAMILY={getattr(main, 'CS_LABEL_FAMILY', None)}")
    data_bundle = load_and_prepare(main, args, logger)

    uniq_days = np.unique(data_bundle["dates"])
    folds = build_forward_folds(uniq_days, train_months=args.train_months,
                                cv_end=args.cv_end, logger=logger)
    if not folds:
        raise RuntimeError("无法构建任何折")

    champ = None
    quick_dir = None
    if args.mode in ("all", "quick"):
        if args.hp_names:
            args.hp_names = [x.strip() for x in args.hp_names.split(",") if x.strip()]
        champ, quick_dir = run_quick(main, data_bundle, folds, args, logger)

    if args.mode in ("all", "full"):
        if args.mode == "full":
            if args.force_pid:
                champ = {"pid": args.force_pid}
            else:
                qd = Path(args.quick_dir) if args.quick_dir else _find_latest_champion_dir(logger)
                if qd is None or not (qd / "champion.json").exists():
                    raise RuntimeError("full 模式需 --quick_dir（或 results/ 下存在历史 quick 目录），或 --force_pid")
                with open(qd / "champion.json", encoding="utf-8") as f:
                    champ = json.load(f)
                quick_dir = qd
                logger.info(f"[FULL] 读取冠军: {champ}（quick_dir={qd}）")
        # [REUSE 2026-09-22] 复用源：显式 quick_dir 只验它；否则自动扫描（需全过协议/指纹校验）
        reuse_dir = None
        if getattr(args, "no_reuse", False):
            logger.info("[REUSE] --no_reuse：全部折重训")
        else:
            reuse_dir = _find_reusable_quick_dir(champ["pid"], data_bundle, logger,
                                                 explicit=str(quick_dir) if quick_dir else None)
        run_full(main, data_bundle, folds, champ, args, logger, reuse_dir=reuse_dir)


if __name__ == "__main__":
    main_flow(parse_args())