"""
Executes the SHAP analysis standalone (same code as the notebook cells).
Saves all figures and prints outputs that will also appear when the notebook is run.
"""
import os, warnings
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
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

# ── Reproduce preprocessing pipeline ─────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

base_features = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness',
    'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]
mean_cols  = [f + '_mean'  for f in base_features]
se_cols    = [f + '_se'    for f in base_features]
worst_cols = [f + '_worst' for f in base_features]
columns    = ['id', 'diagnosis'] + mean_cols + se_cols + worst_cols
feature_columns = mean_cols + se_cols + worst_cols

df = pd.read_csv('data/wdbc.data', header=None, names=columns)
X  = df[feature_columns].values
y  = (df['diagnosis'] == 'M').astype(int).values

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20,
                                                  random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.20,
                                                    random_state=SEED,
                                                    stratify=y_dev)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_val_sc    = scaler.transform(X_val)
X_test_sc   = scaler.transform(X_test)   # sealed

print(f"Sub-training  : {X_train_sc.shape}")
print(f"Validation    : {X_val_sc.shape}")
print(f"Test (held out): {X_test_sc.shape}  <- NOT used in this section")

# ── Model ─────────────────────────────────────────────────────────────────────
class BreastCancerMLP(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = BreastCancerMLP(input_dim=30)
model.load_state_dict(torch.load('results/baseline_mlp_best.pt',
                                  map_location='cpu'))
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X_val_sc, dtype=torch.float32)).squeeze()
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs >= 0.5).astype(int)

acc = accuracy_score(y_val, preds)
auc = roc_auc_score(y_val, probs)
print(f"\nRestored model — Validation Accuracy : {acc*100:.2f}%")
print(f"Restored model — Validation ROC-AUC  : {auc:.4f}")

# ── SHAP setup ────────────────────────────────────────────────────────────────
def predict_proba_malignant(X_np):
    model.eval()
    with torch.no_grad():
        t      = torch.tensor(np.asarray(X_np, dtype=np.float32))
        logits = model(t).squeeze(-1)
        return torch.sigmoid(logits).numpy()

rng        = np.random.RandomState(SEED)
bg_idx     = rng.choice(len(X_train_sc), size=100, replace=False)
background = X_train_sc[bg_idx]

print("\nBuilding SHAP Explainer ...")
explainer   = shap.Explainer(predict_proba_malignant, background,
                              feature_names=feature_columns)
print("Computing SHAP values for validation set ...")
shap_values = explainer(X_val_sc)

print(f"\nSHAP values shape : {shap_values.values.shape}")
print(f"Base value (mean) : {shap_values.base_values.mean():.4f}")
print("\nAdditivity check (first 3 validation samples):")
for i in range(3):
    pred  = predict_proba_malignant(X_val_sc[i:i+1])[0]
    recon = shap_values.base_values[i] + shap_values.values[i].sum()
    print(f"  Sample {i}: prediction={pred:.4f} | base+SHAP={recon:.4f} | "
          f"diff={abs(pred-recon):.2e}")

# ── Global importance table ───────────────────────────────────────────────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
importance_df['rank'] = importance_df.index + 1

print("\nTop-15 Features by Mean |SHAP| Value:")
print(importance_df.head(15).to_string(index=False))

# ── Bar chart ─────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
top_n  = 15
top_df = importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 7))
colors  = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, top_n))
bars    = ax.barh(top_df['feature'][::-1], top_df['mean_abs_shap'][::-1],
                  color=colors[::-1], edgecolor='white', linewidth=0.5)
ax.set_xlabel('Mean |SHAP Value|  (impact on model output)', fontsize=12)
ax.set_title('Global SHAP Feature Importance\n'
             'Mean absolute SHAP value — Top 15 features\n'
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
print("\nSaved: figures/shap_global_importance.png")

# ── Beeswarm plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False, plot_size=None)
plt.title('SHAP Summary (Beeswarm) Plot\n'
          'Feature impact distribution — Top 15 features\n'
          f'(Validation set, N={X_val_sc.shape[0]})',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('figures/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/shap_summary.png")

# ── Waterfall plots ───────────────────────────────────────────────────────────
representative_cases = [
    (0,  'TN', 'True Negative  — Correct Benign Detection'),
    (3,  'TP', 'True Positive  — Correct Malignancy Detection'),
    (48, 'FN', 'False Negative — Missed Malignancy (Model Error)'),
    (53, 'FP', 'False Positive — Benign Flagged as Malignant (Model Error)'),
]

for val_idx, case_label, case_title in representative_cases:
    true_label = y_val[val_idx]
    pred_prob  = predict_proba_malignant(X_val_sc[val_idx:val_idx+1])[0]
    pred_label = int(pred_prob >= 0.5)
    label_str  = lambda l: 'Malignant' if l == 1 else 'Benign'

    fig, ax = plt.subplots(figsize=(12, 7))
    shap.plots.waterfall(shap_values[val_idx], max_display=12, show=False)
    plt.title(
        f'SHAP Waterfall Plot — Case {case_label}\n'
        f'{case_title}\n'
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

print("\n=== All SHAP figures generated successfully. ===")
print("\nFinal figure inventory:")
for f in sorted(os.listdir('figures')):
    size_kb = os.path.getsize(f'figures/{f}') // 1024
    print(f"  figures/{f}  ({size_kb} KB)")
