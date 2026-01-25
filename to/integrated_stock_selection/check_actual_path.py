import os
from pathlib import Path

def check_directory_contents():
    """检查目录的实际内容"""
    base_path = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily")
    print(f"检查目录: {base_path}")
    print(f"目录存在: {base_path.exists()}")

    if base_path.exists():
        print("目录内容:")
        for item in base_path.iterdir():
            print(f"  {item.name} (目录: {item.is_dir()})")

            # 如果是目录，列出其内容
            if item.is_dir():
                sub_items = list(item.iterdir())[:5]  # 只显示前5个
                print(f"    包含: {[si.name for si in sub_items]}...")

if __name__ == "__main__":
    check_directory_contents()