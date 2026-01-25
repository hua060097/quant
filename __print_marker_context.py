from pathlib import Path
path = Path(r"c:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/data_loader.py")
text = path.read_text(encoding="utf-8")
marker = "            self.logger.error(\"���ظ�������ʧ��: {e}\")\n            return None\n"
start = text.index(marker)
print(text[start:start+len(marker)+10])
