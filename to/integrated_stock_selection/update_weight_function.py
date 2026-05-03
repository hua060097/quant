"""
修改 _calculate_professional_sample_weights 函数，简化参数和逻辑
"""

# 新的简化函数代码
new_function_code = '''    try:
        logger.info("样本权重计算 (简化版) 开始")
        n = len(y_train)
        logger.info(f"  总样本数: {n}")

        # ---------- 基本统计 ----------
        logger.info("标签统计:")
        logger.info(f"  min={np.nanmin(y_train):.6f}, max={np.nanmax(y_train):.6f}, mean={np.nanmean(y_train):.6f}, std={np.nanstd(y_train):.6f}")
        logger.info(f"  NaN count in labels: {int(np.isnan(y_train).sum())}, Inf count: {int(np.isinf(y_train).sum())}")

        # ---------- 基于 is_monster 设置权重 ----------
        # is_monster == 1: 权重为 5.0（优质样本）
        # is_monster == 0 或 None: 权重为 1.0（普通样本）
        monster_weight_multiplier = np.ones(n)  # 默认全部为1.0

        if is_monster is not None:
            is_monster_arr = np.asarray(is_monster, dtype=float).copy()

            # 统计 is_monster 分布
            monster_count = int(np.sum(is_monster_arr == 1.0))
            normal_count = int(np.sum(is_monster_arr == 0.0))
            other_count = len(is_monster_arr) - monster_count - normal_count

            logger.info(f"[is_monster] 统计:")
            logger.info(f"  is_monster==1 (优质样本): {monster_count} ({monster_count/n*100:.2f}%)")
            logger.info(f"  is_monster==0 (普通样本): {normal_count} ({normal_count/n*100:.2f}%)")
            if other_count > 0:
                logger.warning(f"  is_monster 其他值: {other_count} (将视为普通样本)")

            # 设置权重乘数
            monster_mask = (is_monster_arr == 1.0)
            monster_weight_multiplier[monster_mask] = 5.0  # 优质样本权重为5
            # 其他样本（包括 is_monster==0）保持权重为1.0

            logger.info(f"[is_monster] 权重乘数设置:")
            logger.info(f"  优质样本权重乘数: 5.0")
            logger.info(f"  普通样本权重乘数: 1.0")
            logger.info(f"  权重乘数统计: min={monster_weight_multiplier.min():.1f}, max={monster_weight_multiplier.max():.1f}")
        else:
            logger.info("[is_monster] 未提供 is_monster 字段，所有样本使用统一权重乘数 1.0")

        # ---------- 可选：基于 turnover_rate 和 circ_mv 进行微调 ----------
        w_final = monster_weight_multiplier.copy()

        if turnover_rate is not None and circ_mv is not None:
            logger.info("[可选] 基于 turnover_rate 和 circ_mv 进行微调")

            turnover = np.asarray(turnover_rate, dtype=float).copy()
            mcap = np.asarray(circ_mv, dtype=float).copy()

            # 简单处理：将无效值替换为0
            turnover = np.nan_to_num(turnover, nan=0.0, posinf=0.0, neginf=0.0)
            mcap = np.nan_to_num(mcap, nan=0.0, posinf=0.0, neginf=0.0)

            # 标准化到 [0.9, 1.1] 范围的微调因子
            if len(turnover) == n and len(mcap) == n:
                # 计算标准化换手率和市值
                turn_norm = (turnover - np.mean(turnover)) / (np.std(turnover) + 1e-12)
                mcap_norm = (mcap - np.mean(mcap)) / (np.std(mcap) + 1e-12)

                # 组合微调因子（权重较小）
                adjustment_factor = 1.0 + 0.05 * turn_norm + 0.05 * mcap_norm
                adjustment_factor = np.clip(adjustment_factor, 0.9, 1.1)

                # 应用微调
                w_final = w_final * adjustment_factor

                logger.info(f"微调后权重统计: min={np.nanmin(w_final):.4f}, max={np.nanmax(w_final):.4f}, mean={np.nanmean(w_final):.4f}")
            else:
                logger.warning("[WARNING] turnover_rate 或 circ_mv 长度不匹配，跳过微调")
        else:
            logger.info("[可选] 未提供 turnover_rate 和 circ_mv，跳过微调")

        # ---------- 最终统计 ----------
        logger.info("最终权重统计:")
        logger.info(f"  min={float(np.nanmin(w_final)):.4f}, max={float(np.nanmax(w_final)):.4f}, mean={float(np.nanmean(w_final)):.4f}, median={float(np.median(w_final)):.4f}")

        # 确保无 NaN/Inf
        w_final = np.nan_to_num(w_final, nan=1.0, posinf=5.0, neginf=1.0)

        return w_final

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("返回均匀权重作为后备")
        return np.ones(len(y_train) if y_train is not None else 100)'''

# 读取文件
file_path = r'C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\main_alpha原始_v2_parallel_validate_fixed.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到函数定义所在的行号
func_start_line = None
try_line = None

for i, line in enumerate(lines):
    if 'def _calculate_professional_sample_weights(' in line:
        func_start_line = i
        print(f"Found function definition at line {i+1}")
    if func_start_line is not None and '    try:' in line and try_line is None:
        try_line = i
        print(f"Found 'try:' at line {i+1}")
        break

if func_start_line is None or try_line is None:
    print("Error: Could not find function")
    exit(1)

# 找到函数的结束行（下一个def或文件结束）
func_end_line = len(lines)
for i in range(try_line + 1, len(lines)):
    if lines[i].startswith('def '):
        func_end_line = i
        print(f"Found next function at line {i+1}, function ends at line {i}")
        break

print(f"\nWill replace lines {try_line + 1} to {func_end_line}")
print(f"Keeping lines 1 to {try_line + 1}")
print(f"New function has {len(new_function_code.split(chr(10)))} lines")

# 构建新文件
new_lines = (
    lines[:try_line + 1] +  # 保留到 try: 这一行
    [new_function_code + '\n'] +  # 添加新的函数体
    lines[func_end_line:]  # 保留后面的部分
)

# 写入新文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("\n✅ Function updated successfully!")
print(f"Total lines: {len(lines)} -> {len(new_lines)}")
