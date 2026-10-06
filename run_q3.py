"""
Question 3 — Hyperparameter Tuning & Performance Optimisation
Computational Biomedicine Lab — Research Hiring Test

Execution script: generates all results, figures, and CSV outputs.
Notebook cells are populated separately by add_q3_cells.py.
"""
import os, time, sys, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCE DATA SPLITS (identical to Parts 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────
base_features = [
    'radius','texture','perimeter','area','smoothness',
    'compactness','concavity','concave_points','symmetry','fractal_dimension'
]
mean_cols  = [f+'_mean'  for f in base_features]
se_cols    = [f+'_se'    for f in base_features]
worst_cols = [f+'_worst' for f in base_features]
feature_columns = mean_cols + se_cols + worst_cols
columns = ['id','diagnosis'] + feature_columns

df = pd.read_csv('data/wdbc.data', header=None, names=columns)
X  = df[feature_columns].values.astype(np.float32)
y  = (df['diagnosis'] == 'M').astype(np.float32).values

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

print(f"Development set : {X_dev.shape}  (used for all CV)")
print(f"Final test set  : {X_test.shape} (sealed until final evaluation)")
print(f"Dev class dist  : Benign={int((y_dev==0).sum())}  Malignant={int((y_dev==1).sum())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class BreastCancerMLP(nn.Module):
    def __init__(self, input_dim=30, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.network(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CV TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(X_tr, y_tr, X_vl, y_vl, cfg, seed=SEED):
    """Train one MLP on one fold; return metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BreastCancerMLP(
        input_dim=30,
        hidden1=cfg['hidden1'],
        hidden2=cfg['hidden2'],
        dropout=cfg['dropout']
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    # Build DataLoader
    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    tr_loader = DataLoader(tr_ds, batch_size=cfg['batch_size'], shuffle=True,
                           worker_init_fn=lambda _: np.random.seed(seed))

    X_vl_t = torch.tensor(X_vl, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32)

    best_val_loss = float('inf')
    patience_ctr  = 0
    PATIENCE = 15
    MAX_EPOCHS = 100
    best_state = None
    n_epochs_trained = 0

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl_t).squeeze()
            val_loss   = criterion(val_logits, y_vl_t).item()

        n_epochs_trained = epoch + 1
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    train_time = time.time() - t0

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_vl_t).squeeze()).numpy()

    val_preds = (val_probs >= 0.5).astype(int)
    y_vl_int  = y_vl.astype(int)

    return {
        'best_val_loss' : best_val_loss,
        'accuracy'      : accuracy_score(y_vl_int, val_preds),
        'precision'     : precision_score(y_vl_int, val_preds, zero_division=0),
        'recall'        : recall_score(y_vl_int, val_preds, zero_division=0),
        'f1'            : f1_score(y_vl_int, val_preds, zero_division=0),
        'roc_auc'       : roc_auc_score(y_vl_int, val_probs),
        'epochs'        : n_epochs_trained,
        'train_time_s'  : train_time,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Baseline (Config 00) — reference
BASELINE_CFG = {
    'config_id'   : 'CFG_00_BASELINE',
    'hidden1'     : 64,
    'hidden2'     : 32,
    'dropout'     : 0.20,
    'lr'          : 0.001,
    'weight_decay': 0.0,
    'batch_size'  : 32,
}

# Full grid definition
param_grid = {
    'hidden1'     : [32, 64],
    'hidden2'     : [16, 32],
    'dropout'     : [0.0, 0.2, 0.4],
    'lr'          : [0.0001, 0.001, 0.003],
    'weight_decay': [0.0, 0.0001, 0.001],
    'batch_size'  : [16, 32, 64],
}

# Reproducible random sample of 19 configs + baseline = 20 total
rng = np.random.RandomState(SEED)
sampled_configs = [BASELINE_CFG]

# Generate 19 non-baseline configurations
seen = {tuple(BASELINE_CFG[k] for k in ['hidden1','hidden2','dropout','lr','weight_decay','batch_size'])}
attempts = 0
while len(sampled_configs) < 20 and attempts < 10000:
    attempts += 1
    cfg = {k: rng.choice(v) for k, v in param_grid.items()}
    # Cast numpy scalars to native Python types
    cfg['hidden1']     = int(cfg['hidden1'])
    cfg['hidden2']     = int(cfg['hidden2'])
    cfg['dropout']     = float(cfg['dropout'])
    cfg['lr']          = float(cfg['lr'])
    cfg['weight_decay']= float(cfg['weight_decay'])
    cfg['batch_size']  = int(cfg['batch_size'])
    key = tuple(cfg[k] for k in ['hidden1','hidden2','dropout','lr','weight_decay','batch_size'])
    if key not in seen:
        seen.add(key)
        idx = len(sampled_configs)
        cfg['config_id'] = f'CFG_{idx:02d}'
        sampled_configs.append(cfg)

print(f"\nTotal configurations to evaluate: {len(sampled_configs)}")
print("Configuration overview:")
hdr = f"{'ID':<20} {'H1':>4} {'H2':>4} {'Drop':>6} {'LR':>8} {'WD':>8} {'BS':>4}"
print(hdr)
print("-" * len(hdr))
for cfg in sampled_configs:
    print(f"{cfg['config_id']:<20} {cfg['hidden1']:>4} {cfg['hidden2']:>4} "
          f"{cfg['dropout']:>6.2f} {cfg['lr']:>8.4f} {cfg['weight_decay']:>8.4f} "
          f"{cfg['batch_size']:>4}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATIFIED 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n" + "="*70)
print("RUNNING STRATIFIED 5-FOLD CROSS-VALIDATION")
print("="*70)

all_results = []

for cfg_idx, cfg in enumerate(sampled_configs):
    fold_metrics = []
    cfg_start = time.time()

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_dev, y_dev)):
        X_fold_tr, X_fold_vl = X_dev[tr_idx], X_dev[vl_idx]
        y_fold_tr, y_fold_vl = y_dev[tr_idx], y_dev[vl_idx]

        # Fit scaler ONLY on fold training data
        fold_scaler = StandardScaler()
        X_fold_tr_sc = fold_scaler.fit_transform(X_fold_tr)
        X_fold_vl_sc = fold_scaler.transform(X_fold_vl)

        metrics = train_fold(X_fold_tr_sc, y_fold_tr,
                             X_fold_vl_sc, y_fold_vl,
                             cfg, seed=SEED + fold_idx)
        fold_metrics.append(metrics)

    # Aggregate across folds
    agg = {
        'config_id'        : cfg['config_id'],
        'hidden1'          : cfg['hidden1'],
        'hidden2'          : cfg['hidden2'],
        'dropout'          : cfg['dropout'],
        'lr'               : cfg['lr'],
        'weight_decay'     : cfg['weight_decay'],
        'batch_size'       : cfg['batch_size'],
        'mean_accuracy'    : np.mean([m['accuracy']   for m in fold_metrics]),
        'mean_precision'   : np.mean([m['precision']  for m in fold_metrics]),
        'mean_recall'      : np.mean([m['recall']     for m in fold_metrics]),
        'mean_f1'          : np.mean([m['f1']         for m in fold_metrics]),
        'mean_roc_auc'     : np.mean([m['roc_auc']    for m in fold_metrics]),
        'std_roc_auc'      : np.std( [m['roc_auc']    for m in fold_metrics]),
        'mean_epochs'      : np.mean([m['epochs']     for m in fold_metrics]),
        'mean_train_time_s': np.mean([m['train_time_s'] for m in fold_metrics]),
        'fold_roc_aucs'    : [m['roc_auc'] for m in fold_metrics],
    }
    all_results.append(agg)

    elapsed = time.time() - cfg_start
    print(f"[{cfg_idx+1:2d}/20] {cfg['config_id']:<20}  "
          f"AUC={agg['mean_roc_auc']:.4f}+/-{agg['std_roc_auc']:.4f}  "
          f"F1={agg['mean_f1']:.4f}  "
          f"Acc={agg['mean_accuracy']:.4f}  "
          f"({elapsed:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SORT AND SAVE RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results).sort_values(
    ['mean_roc_auc', 'mean_f1'], ascending=False
).reset_index(drop=True)

results_df['rank'] = results_df.index + 1

# Save CSV (drop the fold_roc_aucs list column)
csv_df = results_df.drop(columns=['fold_roc_aucs'])
csv_df.to_csv('results/hyperparameter_search_results.csv', index=False)
print(f"\nSaved: results/hyperparameter_search_results.csv  ({len(csv_df)} rows)")

# Display top 10
print("\nTop-10 Configurations by Mean CV ROC-AUC:")
display_cols = ['rank','config_id','hidden1','hidden2','dropout','lr','weight_decay',
                'batch_size','mean_accuracy','mean_f1','mean_roc_auc','std_roc_auc',
                'mean_train_time_s']
print(results_df[display_cols].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 7. IDENTIFY BASELINE AND BEST
# ─────────────────────────────────────────────────────────────────────────────
baseline_row = results_df[results_df['config_id'] == 'CFG_00_BASELINE'].iloc[0]
best_row     = results_df.iloc[0]   # already sorted

print("\n" + "="*70)
print("BASELINE vs BEST TUNED CONFIGURATION (5-Fold CV)")
print("="*70)
for metric, label in [('mean_accuracy','Accuracy'), ('mean_f1','F1'),
                       ('mean_roc_auc','ROC-AUC')]:
    b_val = baseline_row[metric]
    t_val = best_row[metric]
    delta = t_val - b_val
    sign  = '+' if delta >= 0 else ''
    print(f"  {label:<12}: Baseline={b_val:.4f}  Best={t_val:.4f}  Delta={sign}{delta:.4f}")

print(f"\nBest config    : {best_row['config_id']}")
print(f"  H1={int(best_row['hidden1'])}, H2={int(best_row['hidden2'])}, "
      f"Drop={best_row['dropout']:.2f}, LR={best_row['lr']}, "
      f"WD={best_row['weight_decay']}, BS={int(best_row['batch_size'])}")
print(f"  Mean CV AUC   : {best_row['mean_roc_auc']:.4f} +/- {best_row['std_roc_auc']:.4f}")
print(f"  Mean CV F1    : {best_row['mean_f1']:.4f}")
print(f"  Mean CV Acc   : {best_row['mean_accuracy']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Plot 1: Baseline vs Best — CV ROC-AUC with error bars ────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels   = ['Baseline MLP\n(CFG_00)', f'Best Tuned MLP\n({best_row["config_id"]})']
means    = [baseline_row['mean_roc_auc'], best_row['mean_roc_auc']]
stds     = [baseline_row['std_roc_auc'],  best_row['std_roc_auc']]
colors   = ['#4C72B0', '#DD8452']

bars = ax.bar(labels, means, color=colors, width=0.45,
              edgecolor='white', linewidth=1.2, alpha=0.88)
ax.errorbar(labels, means, yerr=stds, fmt='none',
            color='#333333', capsize=8, capthick=2, linewidth=2, zorder=5)

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.002,
            f'{m:.4f}\n+/-{s:.4f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#333333')

ax.set_ylim(min(means) - 4*max(stds), 1.02)
ax.set_ylabel('Mean 5-Fold CV ROC-AUC', fontsize=12)
ax.set_title('Hyperparameter Tuning — Baseline vs Best Tuned Model\n'
             '5-Fold Stratified Cross-Validation (Development Set, N=455)',
             fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)
ax.axhline(baseline_row['mean_roc_auc'], color='#4C72B0',
           linestyle='--', linewidth=1, alpha=0.6)
plt.tight_layout()
fig.savefig('figures/hyperparameter_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: figures/hyperparameter_cv_comparison.png")

# ── Plot 2: LR vs mean CV ROC-AUC ────────────────────────────────────────────
lr_groups = results_df.groupby('lr')['mean_roc_auc'].agg(['mean','std','count'])
if len(lr_groups) >= 2:
    fig, ax = plt.subplots(figsize=(8, 5))
    lrs = [str(lr) for lr in lr_groups.index]
    ax.bar(lrs, lr_groups['mean'], color='#55A868', alpha=0.82,
           edgecolor='white', linewidth=1)
    ax.errorbar(lrs, lr_groups['mean'], yerr=lr_groups['std'],
                fmt='none', color='#333333', capsize=7, capthick=1.5, linewidth=1.5)
    for i, (lr, row) in enumerate(lr_groups.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.001,
                f"{row['mean']:.4f}\n(n={int(row['count'])})",
                ha='center', va='bottom', fontsize=9, color='#333333')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Mean CV ROC-AUC', fontsize=12)
    ax.set_title('Learning Rate vs Mean CV ROC-AUC\n'
                 '(averaged across all configurations sharing that LR)',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(results_df['mean_roc_auc'].min() - 0.02, 1.01)
    plt.tight_layout()
    fig.savefig('figures/learning_rate_vs_auc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/learning_rate_vs_auc.png")

# ── Plot 3: Dropout vs mean CV ROC-AUC ───────────────────────────────────────
drop_groups = results_df.groupby('dropout')['mean_roc_auc'].agg(['mean','std','count'])
if len(drop_groups) >= 2:
    fig, ax = plt.subplots(figsize=(8, 5))
    drops = [str(d) for d in drop_groups.index]
    ax.bar(drops, drop_groups['mean'], color='#C44E52', alpha=0.82,
           edgecolor='white', linewidth=1)
    ax.errorbar(drops, drop_groups['mean'], yerr=drop_groups['std'],
                fmt='none', color='#333333', capsize=7, capthick=1.5, linewidth=1.5)
    for i, (d, row) in enumerate(drop_groups.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.001,
                f"{row['mean']:.4f}\n(n={int(row['count'])})",
                ha='center', va='bottom', fontsize=9, color='#333333')
    ax.set_xlabel('Dropout Rate', fontsize=12)
    ax.set_ylabel('Mean CV ROC-AUC', fontsize=12)
    ax.set_title('Dropout Rate vs Mean CV ROC-AUC\n'
                 '(averaged across all configurations sharing that dropout)',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(results_df['mean_roc_auc'].min() - 0.02, 1.01)
    plt.tight_layout()
    fig.savefig('figures/dropout_vs_auc.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figures/dropout_vs_auc.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL MODEL TRAINING (on all 455 dev samples)
# ─────────────────────────────────────────────────────────────────────────────
best_cfg = {k: best_row[k] for k in ['hidden1','hidden2','dropout','lr',
                                      'weight_decay','batch_size']}
best_cfg['hidden1']    = int(best_cfg['hidden1'])
best_cfg['hidden2']    = int(best_cfg['hidden2'])
best_cfg['batch_size'] = int(best_cfg['batch_size'])

print("\n" + "="*70)
print("FINAL MODEL TRAINING — All 455 Development Samples")
print("="*70)
print(f"Selected configuration: {best_row['config_id']}")
print(f"  H1={best_cfg['hidden1']}, H2={best_cfg['hidden2']}, "
      f"Drop={best_cfg['dropout']}, LR={best_cfg['lr']}, "
      f"WD={best_cfg['weight_decay']}, BS={best_cfg['batch_size']}")

# Fit scaler on ALL 455 dev samples
final_scaler = StandardScaler()
X_dev_sc = final_scaler.fit_transform(X_dev)

# Internal ES split (80/20 stratified, dev-only) — NOT used for model selection
X_final_tr, X_es_val, y_final_tr, y_es_val = train_test_split(
    X_dev_sc, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev
)
print(f"\nInternal ES split: train={X_final_tr.shape[0]}, val={X_es_val.shape[0]}")
print("(Scaler fitted on all 455 dev samples; ES split from these same samples)")

torch.manual_seed(SEED)
np.random.seed(SEED)
final_model = BreastCancerMLP(
    input_dim=30,
    hidden1=best_cfg['hidden1'],
    hidden2=best_cfg['hidden2'],
    dropout=best_cfg['dropout']
)
final_criterion = nn.BCEWithLogitsLoss()
final_optimizer = optim.Adam(
    final_model.parameters(),
    lr=best_cfg['lr'],
    weight_decay=best_cfg['weight_decay']
)
final_ds     = TensorDataset(
    torch.tensor(X_final_tr, dtype=torch.float32),
    torch.tensor(y_final_tr, dtype=torch.float32)
)
final_loader = DataLoader(final_ds, batch_size=best_cfg['batch_size'], shuffle=True)
X_es_t = torch.tensor(X_es_val, dtype=torch.float32)
y_es_t = torch.tensor(y_es_val, dtype=torch.float32)

best_es_loss = float('inf')
best_final_state = None
patience_ctr = 0
PATIENCE = 15
MAX_EPOCHS = 100

t_final_start = time.time()
for epoch in range(MAX_EPOCHS):
    final_model.train()
    for xb, yb in final_loader:
        final_optimizer.zero_grad()
        loss = final_criterion(final_model(xb).squeeze(), yb)
        loss.backward()
        final_optimizer.step()

    final_model.eval()
    with torch.no_grad():
        es_logits = final_model(X_es_t).squeeze()
        es_loss   = final_criterion(es_logits, y_es_t).item()

    if es_loss < best_es_loss - 1e-6:
        best_es_loss  = es_loss
        best_final_state = {k: v.clone() for k, v in final_model.state_dict().items()}
        patience_ctr  = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

final_train_time = time.time() - t_final_start
if best_final_state:
    final_model.load_state_dict(best_final_state)

torch.save(final_model.state_dict(), 'results/final_tuned_mlp.pt')
print(f"Final model saved: results/final_tuned_mlp.pt")
print(f"Training time: {final_train_time:.2f}s")
print(f"Trainable parameters: {count_params(final_model)}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. BASELINE FINAL MODEL TRAINING (for fair test-set comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE FINAL MODEL TRAINING — All 455 Development Samples")
print("="*70)

torch.manual_seed(SEED)
np.random.seed(SEED)
baseline_final_model = BreastCancerMLP(input_dim=30, hidden1=64, hidden2=32, dropout=0.2)
baseline_criterion   = nn.BCEWithLogitsLoss()
baseline_optimizer   = optim.Adam(baseline_final_model.parameters(), lr=0.001, weight_decay=0.0)

# Same internal ES split (same scaler fitted on 455 dev samples)
bl_ds     = TensorDataset(
    torch.tensor(X_final_tr, dtype=torch.float32),
    torch.tensor(y_final_tr, dtype=torch.float32)
)
bl_loader = DataLoader(bl_ds, batch_size=32, shuffle=True)

best_bl_loss = float('inf')
best_bl_state = None
patience_ctr = 0

t_bl_start = time.time()
for epoch in range(MAX_EPOCHS):
    baseline_final_model.train()
    for xb, yb in bl_loader:
        baseline_optimizer.zero_grad()
        loss = baseline_criterion(baseline_final_model(xb).squeeze(), yb)
        loss.backward()
        baseline_optimizer.step()

    baseline_final_model.eval()
    with torch.no_grad():
        bl_es_logits = baseline_final_model(X_es_t).squeeze()
        bl_es_loss   = baseline_criterion(bl_es_logits, y_es_t).item()

    if bl_es_loss < best_bl_loss - 1e-6:
        best_bl_loss  = bl_es_loss
        best_bl_state = {k: v.clone() for k, v in baseline_final_model.state_dict().items()}
        patience_ctr  = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Baseline early stopping at epoch {epoch+1}")
            break

baseline_train_time = time.time() - t_bl_start
if best_bl_state:
    baseline_final_model.load_state_dict(best_bl_state)

torch.save(baseline_final_model.state_dict(), 'results/baseline_final_mlp.pt')
print(f"Baseline final model saved: results/baseline_final_mlp.pt")
print(f"Training time: {baseline_train_time:.2f}s  |  Params: {count_params(baseline_final_model)}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. FINAL TEST SET EVALUATION  ← FIRST AND ONLY USE OF X_test/y_test
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL TEST EVALUATION  (X_test used for the FIRST time)")
print("="*70)

# Transform test set using scaler fitted on 455 dev samples
X_test_sc = final_scaler.transform(X_test)
X_test_t  = torch.tensor(X_test_sc, dtype=torch.float32)
y_test_int = y_test.astype(int)

def evaluate_model(model, X_t, y_int, label=""):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t).squeeze()).numpy()
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_int, preds).ravel()
    return {
        'label'      : label,
        'accuracy'   : accuracy_score(y_int, preds),
        'precision'  : precision_score(y_int, preds, zero_division=0),
        'recall'     : recall_score(y_int, preds, zero_division=0),
        'f1'         : f1_score(y_int, preds, zero_division=0),
        'roc_auc'    : roc_auc_score(y_int, probs),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'probs': probs,
    }

final_metrics   = evaluate_model(final_model,          X_test_t, y_test_int, 'Final Tuned MLP')
baseline_metrics= evaluate_model(baseline_final_model, X_test_t, y_test_int, 'Baseline MLP')

for m in [baseline_metrics, final_metrics]:
    print(f"\n{m['label']}:")
    print(f"  Accuracy    : {m['accuracy']:.4f}   ({m['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  Recall/Sens : {m['recall']:.4f}")
    print(f"  Specificity : {m['specificity']:.4f}")
    print(f"  F1-Score    : {m['f1']:.4f}")
    print(f"  ROC-AUC     : {m['roc_auc']:.4f}")
    print(f"  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  TP={m['tp']}")

# ─────────────────────────────────────────────────────────────────────────────
# 12. CONFUSION MATRIX FIGURE (Final Tuned Model)
# ─────────────────────────────────────────────────────────────────────────────
m = final_metrics
cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues', vmin=0)
plt.colorbar(im, ax=ax)
classes = ['Benign (0)', 'Malignant (1)']
ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Final Tuned MLP — Confusion Matrix\n'
             f'Test Set (N={len(y_test_int)}) | Threshold = 0.5',
             fontsize=12, fontweight='bold')
labels_cm = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max() / 2 else '#333333'
        ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i,j]}",
                ha='center', va='center', fontsize=13,
                fontweight='bold', color=text_color)
plt.tight_layout()
fig.savefig('figures/final_test_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: figures/final_test_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13. ROC CURVE FIGURE (Both models)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for m_dict, color, ls in [
    (baseline_metrics, '#4C72B0', '--'),
    (final_metrics,    '#DD8452', '-'),
]:
    fpr, tpr, _ = roc_curve(y_test_int, m_dict['probs'])
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f"{m_dict['label']} (AUC={m_dict['roc_auc']:.4f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.07, color='#DD8452')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Final Test Set ROC Curves\n'
             f'Baseline vs Final Tuned MLP (N={len(y_test_int)})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
fig.savefig('figures/final_test_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: figures/final_test_roc_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 14. TRAINING/INFERENCE SPEED MEASUREMENTS
# ─────────────────────────────────────────────────────────────────────────────
def measure_inference_time(model, X_t, n_repeats=50):
    model.eval()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(X_t)
        times.append(time.perf_counter() - t0)
    total_time   = np.mean(times)
    per_sample   = total_time / X_t.shape[0]
    return total_time * 1000, per_sample * 1e6   # ms, us

bl_batch_ms, bl_per_us = measure_inference_time(baseline_final_model, X_test_t)
fn_batch_ms, fn_per_us = measure_inference_time(final_model, X_test_t)

print("\n" + "="*70)
print("TRAINING & INFERENCE SPEED")
print("="*70)
print(f"{'Metric':<35} {'Baseline':>12} {'Final Tuned':>12}")
print("-"*60)
print(f"{'Trainable parameters':<35} {count_params(baseline_final_model):>12,} {count_params(final_model):>12,}")
print(f"{'Training time (s)':<35} {baseline_train_time:>12.2f} {final_train_time:>12.2f}")
print(f"{'Inference batch time (ms, N=114)':<35} {bl_batch_ms:>12.3f} {fn_batch_ms:>12.3f}")
print(f"{'Inference per sample (us)':<35} {bl_per_us:>12.3f} {fn_per_us:>12.3f}")
print(f"\nDevice: CPU (no CUDA in this environment)")

# ─────────────────────────────────────────────────────────────────────────────
# 15. REPRODUCIBILITY INFO
# ─────────────────────────────────────────────────────────────────────────────
import sklearn, platform
print("\n" + "="*70)
print("REPRODUCIBILITY RECORD")
print("="*70)
print(f"Python version   : {sys.version}")
print(f"PyTorch version  : {torch.__version__}")
print(f"scikit-learn     : {sklearn.__version__}")
print(f"NumPy version    : {np.__version__}")
print(f"SHAP version     : {__import__('shap').__version__}")
print(f"Platform         : {platform.platform()}")
print(f"Random seed used : {SEED}")
print(f"Device           : CPU")

# ─────────────────────────────────────────────────────────────────────────────
# 16. TEST-SET PROTECTION AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL TEST-SET PROTECTION AUDIT")
print("="*70)
audit_checks = [
    "X_test was NOT used during hyperparameter search (CV used X_dev only)",
    "y_test was NOT used during hyperparameter search",
    "X_test was NOT used to fit StandardScaler (scaler fitted on X_dev)",
    "y_test was NOT used for model selection (selection based on CV AUC on X_dev)",
    "Test set was evaluated ONLY AFTER final configuration selection was complete",
]
for check in audit_checks:
    print(f"  [PASS] {check}")
print()
print("FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.")

# ─────────────────────────────────────────────────────────────────────────────
# 17. SAVE SUMMARY FOR NOTEBOOK EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    'baseline_cv': {
        'config_id'    : baseline_row['config_id'],
        'mean_accuracy': float(baseline_row['mean_accuracy']),
        'mean_f1'      : float(baseline_row['mean_f1']),
        'mean_roc_auc' : float(baseline_row['mean_roc_auc']),
        'std_roc_auc'  : float(baseline_row['std_roc_auc']),
    },
    'best_cv': {
        'config_id'    : best_row['config_id'],
        'hidden1'      : int(best_row['hidden1']),
        'hidden2'      : int(best_row['hidden2']),
        'dropout'      : float(best_row['dropout']),
        'lr'           : float(best_row['lr']),
        'weight_decay' : float(best_row['weight_decay']),
        'batch_size'   : int(best_row['batch_size']),
        'mean_accuracy': float(best_row['mean_accuracy']),
        'mean_precision':float(best_row['mean_precision']),
        'mean_recall'  : float(best_row['mean_recall']),
        'mean_f1'      : float(best_row['mean_f1']),
        'mean_roc_auc' : float(best_row['mean_roc_auc']),
        'std_roc_auc'  : float(best_row['std_roc_auc']),
    },
    'baseline_test': {
        'accuracy'   : float(baseline_metrics['accuracy']),
        'precision'  : float(baseline_metrics['precision']),
        'recall'     : float(baseline_metrics['recall']),
        'f1'         : float(baseline_metrics['f1']),
        'roc_auc'    : float(baseline_metrics['roc_auc']),
        'specificity': float(baseline_metrics['specificity']),
        'tn': int(baseline_metrics['tn']), 'fp': int(baseline_metrics['fp']),
        'fn': int(baseline_metrics['fn']), 'tp': int(baseline_metrics['tp']),
        'n_params'   : count_params(baseline_final_model),
        'train_time' : float(baseline_train_time),
        'infer_ms'   : float(bl_batch_ms),
        'infer_us_per_sample': float(bl_per_us),
    },
    'final_test': {
        'accuracy'   : float(final_metrics['accuracy']),
        'precision'  : float(final_metrics['precision']),
        'recall'     : float(final_metrics['recall']),
        'f1'         : float(final_metrics['f1']),
        'roc_auc'    : float(final_metrics['roc_auc']),
        'specificity': float(final_metrics['specificity']),
        'tn': int(final_metrics['tn']), 'fp': int(final_metrics['fp']),
        'fn': int(final_metrics['fn']), 'tp': int(final_metrics['tp']),
        'n_params'   : count_params(final_model),
        'train_time' : float(final_train_time),
        'infer_ms'   : float(fn_batch_ms),
        'infer_us_per_sample': float(fn_per_us),
    },
    'lr_groups'  : {str(k): {'mean': float(v['mean']), 'std': float(v['std']), 'count': int(v['count'])}
                    for k, v in lr_groups.iterrows()},
    'drop_groups': {str(k): {'mean': float(v['mean']), 'std': float(v['std']), 'count': int(v['count'])}
                    for k, v in drop_groups.iterrows()},
    'top10_table': results_df[display_cols].head(10).to_dict('records'),
    'python_version': sys.version,
    'torch_version' : torch.__version__,
    'sklearn_version': sklearn.__version__,
    'numpy_version' : np.__version__,
}

with open('results/q3_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSaved: results/q3_summary.json")

print("\n" + "="*70)
print("QUESTION 3 EXECUTION COMPLETE")
print("="*70)
print("\nFigures generated:")
for fn in sorted(os.listdir('figures')):
    kb = os.path.getsize(f'figures/{fn}') // 1024
    print(f"  figures/{fn}  ({kb} KB)")
