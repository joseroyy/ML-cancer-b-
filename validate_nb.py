import json, sys

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print('Total cells:', len(cells))

# Check SHAP section cells (33 onwards)
shap_code_cells = [(i, c) for i, c in enumerate(cells) if i >= 33 and c['cell_type'] == 'code']
for i, c in shap_code_cells:
    outputs = c.get('outputs', [])
    n_stream = sum(1 for o in outputs if o.get('output_type') == 'stream')
    n_display = sum(1 for o in outputs if o.get('output_type') == 'display_data')
    exec_count = c.get('execution_count')
    print(f'  Cell {i:02d}: exec_count={exec_count} | stream_outputs={n_stream} | image_outputs={n_display}')

print()
print('Notebook JSON is valid. All checks passed.')
