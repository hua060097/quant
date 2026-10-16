# _archive（冻结快照，非生产路径）

本目录存放已脱离 V4.3 主流水线的历史脚本（零删除原则：移动归档，不删除）。
主目录任何生产脚本均不 import 本目录文件（已核查）。

| 文件 | 原用途 | 替代者 |
| --- | --- | --- |
| s8_state.py | v2.8 手工状态机 | materialize_interactions.py + export_lgb_dataset.py |
| ix_build.py | 旧手工交互构造器 | f1_batch/f1_sector（sector map 改走 gates.load_sector_map，同值已验证） |
| confirm_candidate.py | 单一门控一次性确认 | 统一交互管线 |
| gate_search_prepare_f1.py | 旧胶水脚本 | run_gate_pipeline.py / select_specialists.py |
| cond_supplement.py | 通道 C 事后补捞 | build_top.habitat_seats + select_specialists.py |
| cond_regime_ic.py | 早期状态分组诊断 | gate_search_space.py（部分逻辑吸收） |
| corr_compare.py | 因子值/IC 相关辨析 | 结论固化（r=+0.926 报告） |
| experiments_log_ab.py | log 政策 A/B 一次性实验 | 结论固化 |
| pool_v2.py | 旧版门控池草稿 | build_gate_bank.py |
| h2_rerun.py | 存量因子 H2 重跑 | 结论固化（PLAN §16.20） |
| stability_diag.py | 时间稳定性诊断 | 结论为负，不加门（PLAN §16.20） |

注意：直接运行本目录脚本需 `PYTHONPATH=..`（其内部 `sys.path` 指向自身目录，
跨模块 import 会失败）。如需复活某文件，git mv 回主目录即可（历史完整保留）。
