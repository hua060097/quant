import json

with open(r'C:\Users\86151\.trae-cn\pythons\path\to\local_data\market_status\daily\market_status.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

sample = data.get('20230601', {})
print('Top-level keys:', list(sample.keys()))

def find_key(obj, key_path='', target='daily_change_pct'):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target:
                print(f'Found at: {key_path}.{k} = {v}')
            find_key(v, f'{key_path}.{k}' if key_path else k)

print('\nSearching for daily_change_pct...')
find_key(sample, 'root')

# Also check if there's a simpler structure
print('\n\nChecking if daily_change_pct is at top level...')
if 'daily_change_pct' in sample:
    print(f"Found at root: {sample['daily_change_pct']}")
