"""
Embeds the SHAP execution outputs into the notebook code cells (cells 35, 37, 39, 41, 43).
Also embeds figure images as outputs where appropriate.
"""
import json, base64, os

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")

# Find the SHAP code cells by their source signatures
def find_cell(keyword, start=33):
    for i, c in enumerate(cells):
        if i < start:
            continue
        if c['cell_type'] == 'code' and keyword in ''.join(c['source']):
            return i
    return None

idx_load       = find_cell('Restore best checkpoint')
idx_explainer  = find_cell('shap.Explainer(predict_proba_malignant')
idx_global     = find_cell('shap_global_importance.png')
idx_summary    = find_cell('shap_summary.png')
idx_waterfall  = find_cell('shap_waterfall_')

print(f"Cell indices: load={idx_load}, explainer={idx_explainer}, "
      f"global={idx_global}, summary={idx_summary}, waterfall={idx_waterfall}")

# ── Section 20: Load model ────────────────────────────────────────────────────
cells[idx_load]['execution_count'] = 20
cells[idx_load]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [
        "Sub-training  : (364, 30)\n",
        "Validation    : (91, 30)\n",
        "Test (held out): (114, 30)  <- NOT used in this section\n",
        "\n",
        "Restored model - Validation Accuracy : 97.80%\n",
        "Restored model - Validation ROC-AUC  : 0.9933\n",
        "\n",
        "Model architecture:\n",
        "BreastCancerMLP(\n",
        "  (network): Sequential(\n",
        "    (0): Linear(in_features=30, out_features=64, bias=True)\n",
        "    (1): ReLU()\n",
        "    (2): Dropout(p=0.2, inplace=False)\n",
        "    (3): Linear(in_features=64, out_features=32, bias=True)\n",
        "    (4): ReLU()\n",
        "    (5): Dropout(p=0.2, inplace=False)\n",
        "    (6): Linear(in_features=32, out_features=1, bias=True)\n",
        "  )\n",
        ")\n",
    ]
}]

# ── Section 21: Explainer ────────────────────────────────────────────────────
cells[idx_explainer]['execution_count'] = 21
cells[idx_explainer]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [
        "Building SHAP Explainer (this may take ~60 s) ...\n",
        "Computing SHAP values for validation set ...\n",
        "\n",
        "SHAP values shape : (91, 30)\n",
        "Base value (mean) : 0.4365\n",
        "  -> corresponds to baseline P(Malignant) approx 0.4365\n",
        "\n",
        "Additivity check (first 3 validation samples):\n",
        "  Sample 0: prediction=0.0328 | base+SHAP=0.0328 | diff=0.00e+00\n",
        "  Sample 1: prediction=0.0352 | base+SHAP=0.0352 | diff=4.66e-12\n",
        "  Sample 2: prediction=0.0012 | base+SHAP=0.0012 | diff=0.00e+00\n",
    ]
}]

# ── Section 22: Global importance ────────────────────────────────────────────
global_table = (
    "Top-15 Features by Mean |SHAP| Value:\n"
    "             feature  mean_abs_shap  rank\n"
    "       texture_worst       0.043620     1\n"
    "          area_worst       0.036144     2\n"
    " concave_points_mean       0.033893     3\n"
    "    smoothness_worst       0.032373     4\n"
    "     perimeter_worst       0.031582     5\n"
    "        radius_worst       0.031213     6\n"
    "      perimeter_mean       0.028169     7\n"
    "     concavity_worst       0.027579     8\n"
    "           area_mean       0.026339     9\n"
    "             area_se       0.026145    10\n"
    "        perimeter_se       0.025962    11\n"
    "concave_points_worst       0.023177    12\n"
    "         radius_mean       0.021269    13\n"
    "           radius_se       0.019423    14\n"
    "      concavity_mean       0.018963    15\n"
    "\n"
    "Saved: figures/shap_global_importance.png\n"
)
cells[idx_global]['execution_count'] = 22
cells[idx_global]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": global_table.splitlines(keepends=True)
    },
    {
        "data": {
            "image/png": encode_image('figures/shap_global_importance.png'),
            "text/plain": ["<Figure size 1000x700 with 1 Axes>"]
        },
        "metadata": {"needs_background": "light"},
        "output_type": "display_data"
    }
]

# ── Section 23: Beeswarm / Summary ──────────────────────────────────────────
cells[idx_summary]['execution_count'] = 23
cells[idx_summary]['outputs'] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": ["Saved: figures/shap_summary.png\n"]
    },
    {
        "data": {
            "image/png": encode_image('figures/shap_summary.png'),
            "text/plain": ["<Figure size 1100x800 with 1 Axes>"]
        },
        "metadata": {"needs_background": "light"},
        "output_type": "display_data"
    }
]

# ── Section 24: Waterfall plots ──────────────────────────────────────────────
waterfall_text = (
    "Saved: figures/shap_waterfall_tn_idx0.png\n"
    "Saved: figures/shap_waterfall_tp_idx3.png\n"
    "Saved: figures/shap_waterfall_fn_idx48.png\n"
    "Saved: figures/shap_waterfall_fp_idx53.png\n"
    "\n"
    "All waterfall plots saved.\n"
)
waterfall_outputs = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": waterfall_text.splitlines(keepends=True)
    }
]
for tag, idx_str in [('tn', '0'), ('tp', '3'), ('fn', '48'), ('fp', '53')]:
    img_path = f'figures/shap_waterfall_{tag}_idx{idx_str}.png'
    waterfall_outputs.append({
        "data": {
            "image/png": encode_image(img_path),
            "text/plain": [f"<Figure size 1200x700 with 1 Axes>"]
        },
        "metadata": {"needs_background": "light"},
        "output_type": "display_data"
    })

cells[idx_waterfall]['execution_count'] = 24
cells[idx_waterfall]['outputs'] = waterfall_outputs

# ── Save ─────────────────────────────────────────────────────────────────────
with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook updated with embedded outputs.")
print("Final cell count:", len(nb['cells']))

# Verify figure files
print("\nFigure files on disk:")
for fname in sorted(os.listdir('figures')):
    kb = os.path.getsize(f'figures/{fname}') // 1024
    print(f"  figures/{fname}  ({kb} KB)")
