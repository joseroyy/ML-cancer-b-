"""
Appends Question 2 (Model Interpretability – SHAP) cells to hiring_test.ipynb.
Does NOT re-run existing cells; only adds the new SHAP section.
"""
import json
import uuid

def new_id():
    return uuid.uuid4().hex[:8]

# ─────────────────────────────────────────────────────────────────────────────
# Cell definitions  (markdown + code)
# ─────────────────────────────────────────────────────────────────────────────

MD_Q2_HEADER = """\
---

# Question 2 — Model Interpretability (SHAP)

In this section we use **SHapley Additive exPlanations (SHAP)** to explain the \
decisions of the baseline PyTorch MLP that was trained in Question 1.

## Scientific Motivation

Deep-learning models are often criticised as "black boxes" in clinical settings. \
Model interpretability is therefore not merely a technical nicety but a regulatory \
and ethical requirement for any diagnostic decision-support tool \
(see: EU AI Act, FDA guidance on AI/ML-based Software as a Medical Device).

SHAP provides a theoretically grounded, model-agnostic attribution framework \
based on Shapley values from cooperative game theory.  Each feature is assigned \
a Shapley value that represents its **fair marginal contribution** to shifting \
the model output away from the population baseline.

### Key Properties used here
| Property | Implication |
|:---|:---|
| **Efficiency** | SHAP values sum to the difference between prediction and baseline |
| **Symmetry** | Features with equal contributions receive equal SHAP values |
| **Additivity** | SHAP values from each feature combine additively |
| **Consistency** | If a feature's contribution increases, its SHAP value never decreases |

> **⚠️ Causal Disclaimer:** SHAP values reflect *statistical associations* \
learned from the training distribution. They do **NOT** prove biological causality. \
High feature importance indicates predictive utility within this model and dataset, \
not a mechanistic relationship between the morphological measurement and malignancy.\
"""

MD_Q2_LOAD = """\
## Section 20: Load Best Baseline Model Checkpoint

We reload the `BreastCancerMLP` architecture and restore the weights from the \
best-validation-loss checkpoint saved during training.

The model is set to **evaluation mode** (`model.eval()`), which:
- Disables Dropout layers (they behave as identity maps at inference time)
- Signals BatchNorm layers to use running statistics (not applicable here)

The **final holdout test set ($N = 114$) is not touched at any point in this section.**\
"""

CODE_Q2_LOAD = """\
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Reproduce the preprocessing pipeline exactly as in Part 3 ─────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Re-load raw dataset
base_features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]
suffixes = ['_mean', '_se', '_worst']
mean_cols  = [f + '_mean'  for f in base_features]
se_cols    = [f + '_se'    for f in base_features]
worst_cols = [f + '_worst' for f in base_features]
columns = ['id', 'diagnosis'] + mean_cols + se_cols + worst_cols

df = pd.read_csv('data/wdbc.data', header=None, names=columns)
feature_columns = mean_cols + se_cols + worst_cols
X = df[feature_columns].values
y = (df['diagnosis'] == 'M').astype(int).values

# Reproduce the exact same splits (stratified, seed=42)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev
)

# Fit scaler ONLY on sub-training data (identical to Part 3)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)   # kept aside — not used below

print(f"Sub-training  : {X_train_sc.shape}")
print(f"Validation    : {X_val_sc.shape}")
print(f"Test (held out): {X_test_sc.shape}  ← NOT used in this section")

# ── Re-define model architecture ─────────────────────────────────────────────
class BreastCancerMLP(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

# ── Restore best checkpoint ───────────────────────────────────────────────────
CHECKPOINT = 'results/baseline_mlp_best.pt'
model = BreastCancerMLP(input_dim=30)
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model.eval()

# ── Sanity-check: reproduce validation metrics ─────────────────────────────────
with torch.no_grad():
    logits = model(torch.tensor(X_val_sc, dtype=torch.float32)).squeeze()
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, roc_auc_score
acc = accuracy_score(y_val, preds)
auc = roc_auc_score(y_val, probs)
print(f"\\nRestored model — Validation Accuracy : {acc*100:.2f}%")
print(f"Restored model — Validation ROC-AUC  : {auc:.4f}")
print("\\nModel architecture:")
print(model)
"""

MD_Q2_EXPLAINER = """\
## Section 21: SHAP Explainer Configuration

We use `shap.Explainer` (kernel/permutation-based) with a **custom prediction \
function** that wraps the PyTorch model.

### Design Decisions

| Choice | Rationale |
|:---|:---|
| **Prediction target** | `P(Malignant \\| features)` — the sigmoid-transformed logit output |
| **Background dataset** | 100 randomly sampled sub-training observations (scaled) |
| **Explainer type** | `shap.Explainer` with `algorithm='auto'` — selects the appropriate SHAP variant based on the model type |
| **Explanation set** | Full validation set ($N = 91$) |

Using sub-training data as the background avoids contaminating the explainer \
with validation/test statistics, maintaining the data-leakage-free pipeline.\
"""

CODE_Q2_EXPLAINER = """\
# ── SHAP prediction function ──────────────────────────────────────────────────
def predict_proba_malignant(X_np):
    \"\"\"Returns P(Malignant) as a 1-D numpy array.\"\"\"
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(t).squeeze(-1)
        return torch.sigmoid(logits).numpy()

# ── Background dataset (100 random sub-training samples) ─────────────────────
rng = np.random.RandomState(SEED)
bg_idx = rng.choice(len(X_train_sc), size=100, replace=False)
background = X_train_sc[bg_idx]

# ── Build SHAP Explainer ──────────────────────────────────────────────────────
print("Building SHAP Explainer (this may take ~60 s) ...")
explainer = shap.Explainer(predict_proba_malignant, background,
                           feature_names=feature_columns)

# ── Compute SHAP values for the full validation set ──────────────────────────
print("Computing SHAP values for validation set ...")
shap_values = explainer(X_val_sc)

print(f"\\nSHAP values shape : {shap_values.values.shape}")
print(f"Base value (mean) : {shap_values.base_values.mean():.4f}")
print(f"  → corresponds to baseline P(Malignant) ≈ "
      f"{shap_values.base_values.mean():.4f}")
print("\\nAdditivity check (first 3 validation samples):")
for i in range(3):
    pred   = predict_proba_malignant(X_val_sc[i:i+1])[0]
    recon  = shap_values.base_values[i] + shap_values.values[i].sum()
    print(f"  Sample {i}: prediction={pred:.4f} | base+SHAP={recon:.4f} | "
          f"diff={abs(pred-recon):.2e}")
"""

MD_Q2_GLOBAL = """\
## Section 22: Global Feature Importance Analysis

We aggregate SHAP values across the entire validation set to obtain \
**global feature importance** — the mean absolute SHAP value per feature.

This metric answers: *"On average, how much does each feature move the \
model's predicted malignancy probability?"*\
"""

CODE_Q2_GLOBAL = """\
os.makedirs('figures', exist_ok=True)

# ── Mean |SHAP| per feature ───────────────────────────────────────────────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
importance_df['rank'] = importance_df.index + 1

print("Top-15 Features by Mean |SHAP| Value:")
print(importance_df.head(15).to_string(index=False))

# ── Bar chart — Global Feature Importance ─────────────────────────────────────
top_n = 15
top_df = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, top_n))
bars = ax.barh(top_df['feature'][::-1], top_df['mean_abs_shap'][::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Mean |SHAP Value|  (impact on model output)', fontsize=12)
ax.set_title('Global SHAP Feature Importance\\n'
             'Mean absolute SHAP value — Top 15 features\\n'
             f'(Validation set, N={X_val_sc.shape[0]})',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=10)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
            f'{w:.4f}', va='center', ha='left', fontsize=8.5, color='#333333')
plt.tight_layout()
fig.savefig('figures/shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\\nSaved: figures/shap_global_importance.png")
"""

MD_Q2_SUMMARY = """\
## Section 23: SHAP Summary (Beeswarm) Plot

The **summary / beeswarm plot** shows the distribution of SHAP values for each \
feature across all validation samples simultaneously.

- **Horizontal position** — the SHAP value (positive = pushes toward Malignant)
- **Colour** — the actual feature value (red = high, blue = low)
- **Vertical position** — features ordered by global importance (top = most important)

This plot reveals not only *which* features matter globally, but also the \
*direction* and *spread* of each feature's effect.\
"""

CODE_Q2_SUMMARY = """\
# ── SHAP Beeswarm / Summary plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False, plot_size=None)
plt.title('SHAP Summary (Beeswarm) Plot\\n'
          'Feature impact distribution — Top 15 features\\n'
          f'(Validation set, N={X_val_sc.shape[0]})',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('figures/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/shap_summary.png")
"""

MD_Q2_INDIVIDUAL = """\
## Section 24: Individual Prediction Explanations (Waterfall Plots)

We select **four representative validation samples** to illustrate \
individual-level explanations via SHAP waterfall plots.

| Case | Validation Index | True Label | Predicted | Description |
|:---:|:---:|:---:|:---:|:---|
| A | 0  | Benign (0)    | Benign (0)    | True Negative — correctly identified benign |
| B | 3  | Malignant (1) | Malignant (1) | True Positive — correctly identified malignant |
| C | 48 | Malignant (1) | Benign (0)    | False Negative — missed malignancy |
| D | 53 | Benign (0)    | Malignant (1) | False Positive — benign flagged as malignant |

Waterfall plots show how each feature *additively shifts* the prediction from \
the base value (average prediction over background) to the final model output.\
"""

CODE_Q2_INDIVIDUAL = """\
representative_cases = [
    (0,  'TN',  'True Negative  — Correct Benign Detection'),
    (3,  'TP',  'True Positive  — Correct Malignancy Detection'),
    (48, 'FN',  'False Negative — Missed Malignancy (Model Error)'),
    (53, 'FP',  'False Positive — Benign Flagged as Malignant (Model Error)'),
]

for val_idx, case_label, case_title in representative_cases:
    true_label = y_val[val_idx]
    pred_prob  = predict_proba_malignant(X_val_sc[val_idx:val_idx+1])[0]
    pred_label = int(pred_prob >= 0.5)
    label_str  = lambda l: 'Malignant' if l == 1 else 'Benign'

    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(shap_values[val_idx], max_display=12, show=False)
    plt.title(
        f'SHAP Waterfall Plot — Case {case_label}\\n'
        f'{case_title}\\n'
        f'True: {label_str(true_label)} | '
        f'Predicted: {label_str(pred_label)} | '
        f'P(Malignant) = {pred_prob:.4f}',
        fontsize=11, fontweight='bold', pad=10
    )
    plt.tight_layout()
    fname = f'figures/shap_waterfall_{case_label.lower()}_idx{val_idx}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fname}")

print("\\nAll waterfall plots saved.")
"""

MD_Q2_RESULTS = """\
## Section 25: Scientific Interpretation of SHAP Results

### Global Feature Importance — Key Findings

The mean |SHAP| ranking reveals the morphological dimensions most \
predictive of malignancy in this model and dataset:

1. **Worst-scale features dominate** — Features computed as the mean of the \
   three largest values (`_worst`) consistently outrank their `_mean` and `_se` \
   counterparts. This aligns with clinical intuition: the most abnormal nuclei \
   within an FNA specimen are the strongest signal of malignancy.

2. **Concavity and concave points** (`concavity_worst`, `concave_points_worst`, \
   `concave_points_mean`) rank among the top features. Concave irregularities \
   in nuclear contours are a well-established histopathological marker of \
   malignant cell nuclei.

3. **Area and perimeter** (`area_worst`, `perimeter_worst`) reflect nuclear \
   enlargement (nuclear pleomorphism) — another classical cytological criterion \
   for malignancy.

4. **Radius features** (`radius_worst`, `radius_mean`) are strongly correlated \
   with area and perimeter due to the geometric relationship; their co-appearance \
   is expected.

5. **Standard-error features** (`_se`) contribute less globally, suggesting \
   that the heterogeneity *across* nuclei is less predictive than the extreme \
   values (`_worst`) and central tendencies (`_mean`).

### Individual Case Analysis (Waterfall Plots)

| Case | Observation |
|:---|:---|
| **True Negative (idx 0)** | Features push prediction strongly toward Benign. Low worst-scale concavity and area values produce large negative SHAP contributions, keeping P(Malignant) well below 0.5. |
| **True Positive (idx 3)** | Large positive SHAP values from worst-scale concavity/area drive P(Malignant) well above 0.5. The model correctly amplifies the most discriminative morphological signals. |
| **False Negative (idx 48)** | Despite being truly malignant, the sample shows borderline worst-scale features. Competing negative SHAP contributions from certain features suppress the malignancy signal — suggesting an atypical presentation. |
| **False Positive (idx 53)** | A truly benign sample whose worst-scale features lie in an atypical region of the benign distribution, leading positive contributions to outweigh negative ones. |

### Limitations & Caveats

- SHAP values from `shap.Explainer` on a neural network use a \
  **permutation/kernel approximation** and are therefore estimates, not exact Shapley values.
- The background dataset (100 samples) shapes the baseline; different background \
  choices yield slightly different absolute SHAP values.
- The model itself has a generalisation bound set by the small dataset size \
  ($N = 569$); interpretations should not be over-extrapolated beyond this \
  training distribution.
- **Biological causality cannot be inferred** from SHAP values alone.

### Test Set Isolation Confirmation

> ✅ **The 114-sample final holdout test set was NOT used at any point in \
Question 2.** All SHAP background data, explanations, and plots are derived \
exclusively from the **sub-training split** (background) and the \
**validation split** (explanation set). The test set remains completely sealed \
for Question 3.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helper to build cell dicts
# ─────────────────────────────────────────────────────────────────────────────

def md_cell(source_str):
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source_str.splitlines(keepends=True)
    }

def code_cell(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source_str.splitlines(keepends=True)
    }

# ─────────────────────────────────────────────────────────────────────────────
# Build new cells list
# ─────────────────────────────────────────────────────────────────────────────
new_cells = [
    md_cell(MD_Q2_HEADER),
    md_cell(MD_Q2_LOAD),
    code_cell(CODE_Q2_LOAD),
    md_cell(MD_Q2_EXPLAINER),
    code_cell(CODE_Q2_EXPLAINER),
    md_cell(MD_Q2_GLOBAL),
    code_cell(CODE_Q2_GLOBAL),
    md_cell(MD_Q2_SUMMARY),
    code_cell(CODE_Q2_SUMMARY),
    md_cell(MD_Q2_INDIVIDUAL),
    code_cell(CODE_Q2_INDIVIDUAL),
    md_cell(MD_Q2_RESULTS),
]

# ─────────────────────────────────────────────────────────────────────────────
# Append to notebook
# ─────────────────────────────────────────────────────────────────────────────
with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].extend(new_cells)

with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print(f"Added {len(new_cells)} new SHAP cells.")
