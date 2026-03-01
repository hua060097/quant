
import re

# Read file
with open(r'C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\testaa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line to replace (after the step function, around line 190-210)
pattern = re.compile(r'(# 阶梯函数：根据预测强度分段设置仓位\s*?\s*lead = top1_score - np\.mean\(top5_scores\)\s*?\s*# 将预测值归一化到0-1范围\s*?\s*pred_strength = \(top1_score - top1_min\) \/ \(top1_max - top1_min\) if top1_max > top1_min else 1\s*?\s*# 根据预测强度分段设置仓位（阶梯函数）\s*?\s*if pred_strength >= 0\.02:\s*?# Top 1: 预测强度前2%\s*?\s*weight = 1\.0\s*?\s*elif pred_strength >= 0\.01:\s*?# Top 2: 预测强度前1%\s*?\s*weight = 0\.7\s*?\s*elif pred_strength >= 0\.004:\s*?# Top 3: 预测强度前0\.4%\s*?\s*weight = 0\.4\s*?\s*elif pred_strength >= 0\.007:\s*?# Top 4: 预测强度前0\.7%\s*?\s*weight = 0\.3\s*?\s*else:\s*?# Top 5: 其他（预测强度低于0\.007）\s*?\s*weight = 0\.2\s*?\s*', re.MULTILINE)
)

# Check if pattern was found
if pattern.search(content):
    print(f"Found pattern to replace at lines 194-211")
    # Find the exact match
    match = pattern.search(content)
    if match:
        print(f"Match: {match.group(0)[:100]}")
    else:
        print("Pattern not found")

if not match:
    # Try different pattern
    print("Trying alternative pattern...")
    pattern = re.compile(r'lead = top1_score.*? ? ? ? :')  # More flexible pattern
    if pattern.search(content):
        print(f"Found alternative pattern at lines 194-211")
        match = pattern.search(content)
        if match:
            print(f"Match: {match.group(0)[:100]}")
    else:
        print("Alternative pattern also not found")
"