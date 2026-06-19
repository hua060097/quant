def load(p):
    rows = []
    for line in open(p, encoding='utf-8'):
        parts = line.split()
        if len(parts) >= 4:
            rows.append((parts[0], float(parts[1]), float(parts[2]), int(parts[3])))
    return rows

a = load('C:/Users/86151/AppData/Local/Temp/221_trades.txt')
b = load('C:/Users/86151/AppData/Local/Temp/228_trades.txt')
c = load('C:/Users/86151/AppData/Local/Temp/227_trades.txt')

print(f'{"code":<11} | 221(d)        | 228(d) Δcodex          | 227(d) Δhp')
print('-' * 90)
ttl_codex_pct = 0.0
ttl_hp_pct = 0.0
n_diff_codex = 0
n_diff_hp = 0
for i in range(len(a)):
    code = a[i][0]
    s1, d1 = a[i][2], a[i][3]
    s2, d2 = b[i][2], b[i][3]
    s3, d3 = c[i][2], c[i][3]
    bp = a[i][1]
    rc = (s2 - s1) / bp * 100 if bp else 0.0
    rh = (s3 - s2) / bp * 100 if bp else 0.0
    if s1 != s2 or s2 != s3:
        if s1 != s2:
            n_diff_codex += 1
            ttl_codex_pct += rc
        if s2 != s3:
            n_diff_hp += 1
            ttl_hp_pct += rh
        print(f'{code:<11} | {s1:6.2f}({d1}) | {s2:6.2f}({d2}) {s2-s1:+6.2f} ({rc:+5.2f}%) | {s3:6.2f}({d3}) {s3-s2:+6.2f} ({rh:+5.2f}%)')
print('-' * 90)
print(f'codex 影响：{n_diff_codex} 笔卖价不同，累计相对买价 {ttl_codex_pct:+.2f}%')
print(f'高峰位保护影响：{n_diff_hp} 笔卖价不同，累计相对买价 {ttl_hp_pct:+.2f}%')
