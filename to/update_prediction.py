# 创建预测所有日期的文件
import re
import os

def modify_prediction_function():
    # 读取原始文件
    with open('to/train_sector_models.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式识别并替换特定代码段
    
    # 1. 替换提取最近日期的代码为循环所有日期
    pattern1 = r"(\s+# 初始化预测结果\n\s+predictions = \{\}\n\s+)# 提取最近日期的数据进行预测\n\s+latest_date = predict_features\['date_str'\]\.max\(\)\n\s+latest_features = predict_features\[predict_features\['date_str'\] == latest_date\]\n\s+\n\s+self\.logger\.info\(f\"将使用 \{latest_date\} 的 \{len\(latest_features\)\} 只股票数据进行预测\"\)"
    
    replacement1 = r"\1# 对所有日期的数据进行预测，而不仅仅是最新日期\n        unique_dates = predict_features['date_str'].unique()\n        self.logger.info(f\"将对 {len(unique_dates)} 个交易日的数据进行预测\")"
    
    # 2. 替换处理股票的循环语句
    pattern2 = r"(\s+# 如果有历史数据中的特征重要性信息，优先使用.*?\n.*?feature_weights\[feature\] = importance \* 5  # 调整系数\n\s+)# 处理每只股票\n\s+for _, stock_row in latest_features\.iterrows\(\):"
    
    replacement2 = r"\1# 循环处理每个日期\n        for date_str in unique_dates:\n            date_features = predict_features[predict_features['date_str'] == date_str]\n            self.logger.info(f\"正在预测日期 {date_str} 的 {len(date_features)} 只股票数据\")\n            \n            # 处理每只股票\n            for _, stock_row in date_features.iterrows():"
    
    # 3. 将predictions[stock_code]改为使用唯一的键
    pattern3 = r"(\s+# 将预测结果添加到结果字典\n\s+)predictions\[stock_code\] = \{"
    
    replacement3 = r"\1prediction_key = f\"{stock_code}_{date_str}\"\n            predictions[prediction_key] = {"
    
    # 4. 将'date': latest_date改为'date': date_str
    pattern4 = r"'date': latest_date,"
    
    replacement4 = r"'date': date_str,"
    
    # 应用替换
    content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)
    content = re.sub(pattern4, replacement4, content)
    
    # 保存修改后的文件
    with open('to/train_sector_models_modified.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("文件修改完成，新文件保存为: to/train_sector_models_modified.py")

if __name__ == "__main__":
    modify_prediction_function()
