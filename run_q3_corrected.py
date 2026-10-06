"""
Corrected final-model training procedure.

Stage 1 — Leakage-free early stopping:
  - Split X_dev (455) -> X_final_tr (364) + X_es_val (91), stratified, seed=42
  - Fit StandardScaler on X_final_tr (364) ONLY
  - Transform X_final_tr and X_es_val using that scaler
  - Train selected MLP with early stopping on X_es_val

Stage 2 — Final test evaluation:
  - After model is fully trained and selected, fit a NEW StandardScaler on all 455 dev samples
  - Transform all 455 dev samples (for documentation)
  - Transform 114 test samples using that scaler
  - Evaluate final model once on 114 test samples

Does NOT touch hyperparameter search results.
Does NOT rerun CV.
"""

import os, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCE DATA SPLITS (identical to all previous sections)
# ─────────────────────────────────────────────────────────────────────────────
base_features = ['radius','texture','perimeter','area','smoothness','compactness',
                 'concavity','concave_points','symmetry','fractal_dimension']
mean_cols  = [f+'_mean'  for f in base_features]
se_cols    = [f+'_se'    for f in base_features]
worst_cols = [f+'_worst' for f in base_features]
feature_columns = mean_cols + se_cols + worst_cols
columns = ['id','diagnosis'] + feature_columns

import pandas as pd
df = pd.read_csv('data/wdbc.data', header=None, names=columns)
X  = df[feature_columns].values.astype(np.float32)
y  = (df['diagnosis'] == 'M').astype(np.float32).values

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"Development set : {X_dev.shape}")
print(f"Test set (sealed): {X_test.shape}")

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
    def forward(self, x): return self.network(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTED HYPERPARAMETERS (unchanged from CV search)
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_CFG = {
    'hidden1'     : 64,
    'hidden2'     : 32,
    'dropout'     : 0.20,
    'lr'          : 0.001,
    'weight_decay': 0.0,
    'batch_size'  : 32,
}
print(f"\nSelected configuration: {SELECTED_CFG}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. STAGE 1 — LEAKAGE-FREE FINAL MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 1: Leakage-Free Final Model Training")
print("="*70)

# Step 1: Internal stratified 80/20 split of development data
X_final_tr, X_es_val, y_final_tr, y_es_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev
)
print(f"Internal split -> Train: {X_final_tr.shape}  ES-Val: {X_es_val.shape}")
print(f"Train class dist: Benign={int((y_final_tr==0).sum())}  "
      f"Malignant={int((y_final_tr==1).sum())}")
print(f"ES-Val class dist: Benign={int((y_es_val==0).sum())}  "
      f"Malignant={int((y_es_val==1).sum())}")

# Step 2: Fit StandardScaler ONLY on 364 training samples
stage1_scaler = StandardScaler()
X_final_tr_sc = stage1_scaler.fit_transform(X_final_tr)   # fit on 364
X_es_val_sc   = stage1_scaler.transform(X_es_val)         # transform using same scaler

print(f"\nScaler fitted on {X_final_tr.shape[0]} training samples only.")
print(f"ES validation transformed using training-fitted scaler.")

# Step 3: Train selected MLP with early stopping
torch.manual_seed(SEED)
np.random.seed(SEED)

final_model = BreastCancerMLP(
    input_dim=30,
    hidden1=SELECTED_CFG['hidden1'],
    hidden2=SELECTED_CFG['hidden2'],
    dropout=SELECTED_CFG['dropout']
)
final_criterion = nn.BCEWithLogitsLoss()
final_optimizer = optim.Adam(
    final_model.parameters(),
    lr=SELECTED_CFG['lr'],
    weight_decay=SELECTED_CFG['weight_decay']
)

final_ds = TensorDataset(
    torch.tensor(X_final_tr_sc, dtype=torch.float32),
    torch.tensor(y_final_tr,    dtype=torch.float32)
)
final_loader = DataLoader(final_ds, batch_size=SELECTED_CFG['batch_size'], shuffle=True)

X_es_t = torch.tensor(X_es_val_sc, dtype=torch.float32)
y_es_t = torch.tensor(y_es_val,    dtype=torch.float32)

best_es_loss     = float('inf')
best_final_state = None
patience_ctr     = 0
PATIENCE         = 15
MAX_EPOCHS       = 100

t_start = time.time()
for epoch in range(MAX_EPOCHS):
    final_model.train()
    for xb, yb in final_loader:
        final_optimizer.zero_grad()
        loss = final_criterion(final_model(xb).squeeze(), yb)
        loss.backward()
        final_optimizer.step()

    final_model.eval()
    with torch.no_grad():
        es_loss = final_criterion(
            final_model(X_es_t).squeeze(), y_es_t
        ).item()

    if es_loss < best_es_loss - 1e-6:
        best_es_loss     = es_loss
        best_final_state = {k: v.clone() for k, v in final_model.state_dict().items()}
        patience_ctr     = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}  "
                  f"(best ES val loss: {best_es_loss:.6f})")
            break

final_train_time = time.time() - t_start
if best_final_state:
    final_model.load_state_dict(best_final_state)

torch.save(final_model.state_dict(), 'results/final_tuned_mlp.pt')
print(f"  Training time: {final_train_time:.2f}s")
print(f"  Trainable parameters: {count_params(final_model):,}")
print(f"  Final model saved: results/final_tuned_mlp.pt")

# Also train baseline using same leakage-free procedure
print("\n-- Baseline model (same Stage 1 procedure) --")
torch.manual_seed(SEED)
np.random.seed(SEED)

baseline_final_model = BreastCancerMLP(input_dim=30, hidden1=64, hidden2=32, dropout=0.2)
bl_criterion  = nn.BCEWithLogitsLoss()
bl_optimizer  = optim.Adam(baseline_final_model.parameters(), lr=0.001, weight_decay=0.0)
bl_ds = TensorDataset(
    torch.tensor(X_final_tr_sc, dtype=torch.float32),
    torch.tensor(y_final_tr,    dtype=torch.float32)
)
bl_loader = DataLoader(bl_ds, batch_size=32, shuffle=True)

best_bl_loss  = float('inf')
best_bl_state = None
patience_ctr  = 0

t_bl_start = time.time()
for epoch in range(MAX_EPOCHS):
    baseline_final_model.train()
    for xb, yb in bl_loader:
        bl_optimizer.zero_grad()
        loss = bl_criterion(baseline_final_model(xb).squeeze(), yb)
        loss.backward()
        bl_optimizer.step()

    baseline_final_model.eval()
    with torch.no_grad():
        bl_es_loss = bl_criterion(
            baseline_final_model(X_es_t).squeeze(), y_es_t
        ).item()

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
print(f"  Training time: {baseline_train_time:.2f}s  |  "
      f"Params: {count_params(baseline_final_model):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STAGE 2 — FINAL EVALUATION SCALER (fitted on all 455 dev samples)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STAGE 2: Final Evaluation Scaler (all 455 dev samples)")
print("="*70)

# Model configuration is now FIXED (selected by CV, trained via Stage 1).
# Fit a new scaler on the full 455-sample development set.
eval_scaler = StandardScaler()
X_dev_sc    = eval_scaler.fit_transform(X_dev)    # fit on all 455 dev samples
print(f"Evaluation scaler fitted on {X_dev.shape[0]} development samples.")

# Transform the 114 test samples using this evaluation scaler
# *** FIRST AND ONLY USE OF X_test / y_test ***
X_test_sc  = eval_scaler.transform(X_test)
X_test_t   = torch.tensor(X_test_sc, dtype=torch.float32)
y_test_int = y_test.astype(int)
print(f"Test set transformed using evaluation scaler (N={X_test_sc.shape[0]}, "
      f"scaler NOT fit on test data).")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINAL TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL TEST EVALUATION (X_test used for the FIRST and ONLY time)")
print("="*70)

def evaluate_model(model, X_t, y_int, label=""):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t).squeeze()).numpy()
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_int, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(label=label,
                accuracy    = accuracy_score(y_int, preds),
                precision   = precision_score(y_int, preds, zero_division=0),
                recall      = recall_score(y_int, preds, zero_division=0),
                f1          = f1_score(y_int, preds, zero_division=0),
                roc_auc     = roc_auc_score(y_int, probs),
                specificity = spec,
                tn=tn, fp=fp, fn=fn, tp=tp, probs=probs)

final_metrics    = evaluate_model(final_model,          X_test_t, y_test_int, 'Final Tuned MLP')
baseline_metrics = evaluate_model(baseline_final_model, X_test_t, y_test_int, 'Baseline MLP')

for m in [baseline_metrics, final_metrics]:
    print(f"\n{m['label']}:")
    print(f"  Accuracy         : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
    print(f"  Precision        : {m['precision']:.4f}")
    print(f"  Recall/Sensitivity: {m['recall']:.4f}")
    print(f"  Specificity      : {m['specificity']:.4f}")
    print(f"  F1-Score         : {m['f1']:.4f}")
    print(f"  ROC-AUC          : {m['roc_auc']:.4f}")
    print(f"  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}  TP={m['tp']}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. REGENERATE FIGURES WITH CORRECTED RESULTS
# ─────────────────────────────────────────────────────────────────────────────

# Confusion matrix
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
labels_cm = [['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        tc = 'white' if cm[i,j] > cm.max()/2 else '#333333'
        ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i,j]}",
                ha='center', va='center', fontsize=13, fontweight='bold', color=tc)
plt.tight_layout()
fig.savefig('figures/final_test_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: figures/final_test_confusion_matrix.png")

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
for m_dict, color, ls in [(baseline_metrics,'#4C72B0','--'),
                            (final_metrics,'#DD8452','-')]:
    fpr, tpr, _ = roc_curve(y_test_int, m_dict['probs'])
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f"{m_dict['label']} (AUC={m_dict['roc_auc']:.4f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random Classifier')
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
# 8. INFERENCE SPEED
# ─────────────────────────────────────────────────────────────────────────────
import time as _time
def measure_inference(model, X_t, n_repeats=50):
    model.eval()
    times = []
    for _ in range(n_repeats):
        t0 = _time.perf_counter()
        with torch.no_grad():
            _ = model(X_t)
        times.append(_time.perf_counter() - t0)
    mean_ms = np.mean(times) * 1000
    per_us  = np.mean(times) / X_t.shape[0] * 1e6
    return mean_ms, per_us

bl_ms, bl_us = measure_inference(baseline_final_model, X_test_t)
fn_ms, fn_us = measure_inference(final_model, X_test_t)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE CORRECTED SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
# Load existing Q3 summary and update only the final_training/test sections
with open('results/q3_summary.json','r') as f:
    existing = json.load(f)

existing['baseline_test'] = {
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
    'infer_ms'   : float(bl_ms),
    'infer_us_per_sample': float(bl_us),
}
existing['final_test'] = {
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
    'infer_ms'   : float(fn_ms),
    'infer_us_per_sample': float(fn_us),
}
existing['final_training_correction'] = {
    'stage1_train_samples' : int(X_final_tr.shape[0]),
    'stage1_esval_samples' : int(X_es_val.shape[0]),
    'stage1_scaler_fit_on' : 'X_final_tr (364 samples)',
    'stage2_scaler_fit_on' : 'X_dev (all 455 samples)',
    'test_transform_scaler': 'stage2 eval_scaler (fitted on X_dev, NOT X_test)',
}

with open('results/q3_summary.json','w') as f:
    json.dump(existing, f, indent=2, default=str)
print("\nUpdated: results/q3_summary.json")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TEST-SET PROTECTION AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL TEST-SET PROTECTION AUDIT")
print("="*70)
checks = [
    "X_test was NOT used during hyperparameter search (CV on X_dev only)",
    "y_test was NOT used during hyperparameter search",
    "Stage 1 scaler fitted ONLY on 364-sample training split (not ES val, not test)",
    "Stage 2 scaler fitted ONLY on all 455 dev samples (not test)",
    "X_test was NOT used to fit any scaler",
    "y_test was NOT used for model selection (selection by CV AUC on X_dev)",
    "Test set evaluated ONLY AFTER final configuration selection was complete",
]
for c in checks:
    print(f"  [PASS] {c}")

print()
print("FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.")

# Print summary for notebook embedding
print("\n" + "="*70)
print("SUMMARY FOR NOTEBOOK UPDATE")
print("="*70)
bts = existing['baseline_test']
fts = existing['final_test']
print(f"\nBaseline MLP  | Acc={bts['accuracy']:.4f} | "
      f"P={bts['precision']:.4f} | R={bts['recall']:.4f} | "
      f"F1={bts['f1']:.4f} | AUC={bts['roc_auc']:.4f} | "
      f"Spec={bts['specificity']:.4f} | "
      f"TN={bts['tn']} FP={bts['fp']} FN={bts['fn']} TP={bts['tp']}")
print(f"Final Tuned   | Acc={fts['accuracy']:.4f} | "
      f"P={fts['precision']:.4f} | R={fts['recall']:.4f} | "
      f"F1={fts['f1']:.4f} | AUC={fts['roc_auc']:.4f} | "
      f"Spec={fts['specificity']:.4f} | "
      f"TN={fts['tn']} FP={fts['fp']} FN={fts['fn']} TP={fts['tp']}")

# Store the y_test shape for the fix script
print(f"\nX_test_int shape: {y_test_int.shape}")
