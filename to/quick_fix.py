"""
修复advanced_quant_data.py中的缩进问题
"""

def fix_indentation():
    # 读取文件
    with open('advanced_quant_data.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修复第672行的缩进问题
    if len(lines) > 672:
        if 'batches = ' in lines[671]:
            lines[671] = '        batches = ' + lines[671].strip()[len('batches = '):] + '\n'
    
    # 修复第681-697行的缩进问题
    if len(lines) > 697:
        # 修复完成后的batch_idx以及for future部分
        for i in range(683, 697):
            if 'for future' in lines[i]:
                lines[i] = '            for future in as_completed(future_to_batch_idx):\n'
            elif 'batch_idx =' in lines[i]:
                lines[i] = '                batch_idx = future_to_batch_idx[future]\n'
            elif 'try:' in lines[i]:
                lines[i] = '                try:\n'
            elif 'batch_data =' in lines[i]:
                lines[i] = '                    batch_data = future.result() # This is the dict {field: df}\n'
            elif 'if batch_data:' in lines[i]:
                lines[i] = '                    if batch_data:\n'
            elif 'downloaded_batch_data_results.append' in lines[i]:
                lines[i] = '                        downloaded_batch_data_results.append(batch_data)\n'
            elif 'completed_batches +=' in lines[i]:
                lines[i] = '                        completed_batches += 1\n'
            elif 'logger.info' in lines[i] and 'Downloaded batch' in lines[i]:
                lines[i] = '                    logger.info(f"Downloaded batch {batch_idx+1}/{total_batches} completed. Progress: {completed_batches/total_batches*100:.1f}%")\n'
            elif 'future._result =' in lines[i]:
                lines[i] = '                    future._result = None\n'
            elif 'except Exception' in lines[i]:
                lines[i] = '                except Exception as e:\n'
            elif 'logger.error' in lines[i] and 'Error downloading batch' in lines[i]:
                lines[i] = '                    logger.error(f"Error downloading batch {batch_idx+1}: {e}")\n'
            elif 'logger.debug' in lines[i] and 'traceback' in lines[i]:
                lines[i] = '                    logger.debug(traceback.format_exc())\n'
    
    # 修复try-except块缩进(725-780)
    for i in range(725, 780):
        if i < len(lines):
            if '                try:' in lines[i]:
                lines[i] = '                try:\n'
            elif '                    except Exception as e:' in lines[i]:
                lines[i] = '                except Exception as e:\n'
            elif '                    logger.error' in lines[i] and 'Error distributing data' in lines[i]:
                lines[i] = '                    logger.error(f"Error distributing data for batch {batch_idx+1}: {e}")\n'
    
    # 修复return语句的缩进(875-885)
    for i in range(875, 885):
        if i < len(lines):
            if 'return successful_saves > 0' in lines[i]:
                lines[i] = '        return successful_saves > 0 # Return True if at least one partition was saved successfully\n'
    
    # 修复第940-945行的缩进问题
    if len(lines) > 945:
        for i in range(940, 945):
            if 'if not daily_dir_path.exists()' in lines[i]:
                lines[i] = '        if not daily_dir_path.exists():\n'
            elif 'logger.error' in lines[i] and '日线数据基础目录不存在' in lines[i]:
                lines[i] = '            logger.error(f"日线数据基础目录不存在: {daily_dir_path}")\n'
            elif 'return pd.DataFrame()' in lines[i]:
                lines[i] = '            return pd.DataFrame()\n'
    
    # 修复第960-965行的缩进问题
    if len(lines) > 965:
        for i in range(960, 965):
            if '# 添加股票代码过滤条件' in lines[i]:
                lines[i] = '            # 添加股票代码过滤条件\n'
            elif 'if stock_list:' in lines[i]:
                lines[i] = '            if stock_list:\n'
            elif '# 确保 stock_list 中的元素是字符串类型' in lines[i]:
                lines[i] = '                # 确保 stock_list 中的元素是字符串类型\n'
            elif 'stock_list = [str(s)' in lines[i]:
                lines[i] = '                stock_list = [str(s) for s in stock_list]\n'
            elif '# 检查 stock_code 列是否存在于 schema 中' in lines[i]:
                lines[i] = '                # 检查 stock_code 列是否存在于 schema 中\n'
    
    # 修复第975-985行的缩进问题
    if len(lines) > 985:
        for i in range(975, 985):
            if '# --- 确定读取的列 ---' in lines[i]:
                lines[i] = '            # --- 确定读取的列 ---\n'
            elif 'read_columns = None' in lines[i]:
                lines[i] = '            read_columns = None # 默认读取所有列\n'
            elif 'if fields:' in lines[i]:
                lines[i] = '            if fields:\n'
            elif '# 保证基础列总是被读取' in lines[i]:
                lines[i] = '                # 保证基础列总是被读取，以用于后续处理或排序\n'
            elif 'base_cols = ' in lines[i]:
                lines[i] = '                base_cols = [\'stock_code\', \'date_str\']\n'
            elif '# 确保 fields 中的列也是字符串' in lines[i]:
                lines[i] = '                # 确保 fields 中的列也是字符串\n'
            elif 'fields_str = ' in lines[i]:
                lines[i] = '                fields_str = [str(f) for f in fields]\n'
            elif 'read_columns = list' in lines[i]:
                lines[i] = '                read_columns = list(set(base_cols + fields_str))\n'
    
    # 修复类方法定义的缩进(880-1600)
    for i in range(880, 1600):
        if i < len(lines):
            line = lines[i].strip()
            if line.startswith('def ') and '(self' in line:
                lines[i] = '    ' + line + '\n'
    
    # 修复其他所有return pd.DataFrame()语句的缩进
    for i in range(940, 1065):
        if i < len(lines):
            if 'return pd.DataFrame()' in lines[i]:
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                if current_indent != 12 and current_indent != 16:
                    # 查找上一行的缩进
                    prev_line_indent = 0
                    for j in range(i-1, i-10, -1):
                        if j >= 0 and j < len(lines) and lines[j].strip():
                            prev_line_indent = len(lines[j]) - len(lines[j].lstrip())
                            break
                    lines[i] = ' ' * prev_line_indent + 'return pd.DataFrame()\n'
    
    # 特别修复1025-1035区域的缩进
    for i in range(1025, 1035):
        if i < len(lines):
            if 'else:' in lines[i] and lines[i].strip() == 'else:':
                lines[i] = '            else:\n'
            elif 'logger.warning' in lines[i] and '读取的数据中缺少' in lines[i]:
                lines[i] = '                logger.warning("读取的数据中缺少 \'date_str\' 列，无法创建 \'date\' 列。")\n'
    
    # 修复except语句块的缩进(1045-1065)
    for i in range(1045, 1065):
        if i < len(lines):
            if 'except pa.lib.ArrowObjectNotFound' in lines[i]:
                lines[i] = '        except pa.lib.ArrowObjectNotFound:\n'
            elif 'except pa.lib.ArrowInvalid' in lines[i]:
                lines[i] = '        except pa.lib.ArrowInvalid as e:\n'
            elif 'except KeyError' in lines[i]:
                lines[i] = '        except KeyError as e:\n'
            elif 'except Exception' in lines[i]:
                lines[i] = '        except Exception as e:\n'
            elif 'logger.error' in lines[i] or 'logger.debug' in lines[i]:
                prev_line = lines[i-1].strip() if i-1 < len(lines) else ""
                if prev_line.startswith('except '):
                    lines[i] = '            ' + lines[i].strip() + '\n'
                else:
                    # 保持与上一行相同的缩进
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    lines[i] = ' ' * prev_indent + lines[i].strip() + '\n'
    
    # 写入修复后的文件
    with open('advanced_quant_data.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("缩进问题已修复")

if __name__ == '__main__':
    fix_indentation() 