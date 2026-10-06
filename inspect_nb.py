import json

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c['source'])[:150].replace('\n', ' ')
    print(f'Cell {i:02d}: [{c["cell_type"]}] {src}')
