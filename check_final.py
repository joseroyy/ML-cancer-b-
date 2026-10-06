import os, json

# Notebook cell count
with open('hiring_test.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)
print(f"hiring_test.ipynb : {len(nb['cells'])} cells  ({os.path.getsize('hiring_test.ipynb')//1024} KB)")

# Figures
print("\nfigures/")
for fn in sorted(os.listdir('figures')):
    kb = os.path.getsize(f'figures/{fn}') // 1024
    print(f"  {fn}  ({kb} KB)")

# Results
print("\nresults/")
for fn in sorted(os.listdir('results')):
    kb = os.path.getsize(f'results/{fn}') // 1024
    print(f"  {fn}  ({kb} KB)")
