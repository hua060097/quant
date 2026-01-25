import os
import platform
import shutil
import sqlite3
import time

def get_chrome_profile_path():
    """
    获取当前操作系统的Chrome用户配置主路径。
    """
    system = platform.system()
    if system == "Windows":
        # APPDATA/Local/Google/Chrome/User Data
        return os.path.join(os.getenv('LOCALAPPDATA'), 'Google', 'Chrome', 'User Data')
    elif system == "Darwin": # macOS
        # ~/Library/Application Support/Google/Chrome
        return os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Google', 'Chrome')
    elif system == "Linux":
        # ~/.config/google-chrome
        return os.path.join(os.path.expanduser('~'), '.config', 'google-chrome')
    else:
        raise NotImplementedError(f"不支持的操作系统: {system}")

def clear_chrome_data(profile_path):
    """
    清理指定Chrome配置文件路径下的敏感数据。
    """
    # 确保路径存在
    if not os.path.exists(profile_path):
        print(f"[-] Chrome 配置文件路径不存在: {profile_path}")
        return

    # 通常数据都存放在 'Default' 文件夹下
    default_profile_path = os.path.join(profile_path, 'Default')
    if not os.path.exists(default_profile_path):
        print(f"[-] 'Default' 配置文件不存在于: {default_profile_path}")
        return

    print(f"[+] 找到 Chrome 配置文件: {default_profile_path}")

    # 定义要删除的文件和目录列表
    items_to_delete = {
        # 文件
        "files": [
            "Cookies", "Cookies-journal", "History", "History-journal", 
            "Login Data", "Login Data-journal", "Web Data", "Web Data-journal"
        ],
        # 目录
        "dirs": [
            "Cache", "Code Cache", "GPUCache", "Session Storage", 
            "Local Storage", "IndexedDB", "Application Cache"
        ]
    }

    # --- 开始清理 ---
    print("\n--- 正在清理文件 ---")
    for file_name in items_to_delete["files"]:
        file_path = os.path.join(default_profile_path, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[✓] 已删除文件: {file_path}")
            except OSError as e:
                print(f"[!] 删除文件失败: {file_path} - {e}")
                print("[!] 请确保 Chrome 已完全关闭！")

    print("\n--- 正在清理目录 ---")
    for dir_name in items_to_delete["dirs"]:
        dir_path = os.path.join(default_profile_path, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"[✓] 已删除目录: {dir_path}")
            except OSError as e:
                print(f"[!] 删除目录失败: {dir_path} - {e}")
                print("[!] 请确保 Chrome 已完全关闭！")
                
def main():
    print("=============================================")
    print("=== Chrome 本地数据清理脚本 ===")
    print("=============================================")
    print("[!] 警告：此脚本将删除你的浏览历史、Cookies、缓存和登录会话。")
    print("[!] 在开始前，请务必完全关闭所有 Chrome 浏览器窗口！")
    
    # 倒计时，给用户时间关闭浏览器
    for i in range(5, 0, -1):
        print(f"...{i}", end=' ', flush=True)
        time.sleep(1)
    print("\n")

    try:
        profile_path = get_chrome_profile_path()
        clear_chrome_data(profile_path)
        print("\n[+] 清理操作已完成！")
    except Exception as e:
        print(f"\n[X] 发生未知错误: {e}")

if __name__ == "__main__":
    main()