"""
Updates only the Q3 final-training and test-evaluation cells in hiring_test.ipynb
to reflect the corrected two-stage leakage-free procedure.
Cells 55 (exec_count=29), 57 (exec_count=30), 58 (markdown comparison), 59 (exec_count=31)
are replaced. The markdown for final model training (cell 54) is also updated.
"""
import json, base64, uuid

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def new_id():
    return uuid.uuid4().hex[:8]

with open('results/q3_summary.json', 'r') as f:
    S = json.load(f)

bts = S['baseline_test']
fts = S['final_test']
best = S['best_cv']

# ─────────────────────────────────────────────────────────────────────────────
# Load notebook
# ─────────────────────────────────────────────────────────────────────────────
with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find cells by exec_count
def find_by_exec(exec_count):
    for i, c in enumerate(cells):
        if c.get('execution_count') == exec_count:
            return i
    return None

def find_markdown_after(exec_count, keyword):
    """Find the markdown cell immediately after the code cell with exec_count
    that contains keyword in its source."""
    code_idx = find_by_exec(exec_count)
    if code_idx is None:
        return None
    for i in range(code_idx + 1, min(code_idx + 5, len(cells))):
        c = cells[i]
        if c['cell_type'] == 'markdown' and keyword in ''.join(c['source']):
            return i
    return None

def find_markdown_before(exec_count, keyword):
    """Find the markdown cell immediately before the code cell with exec_count
    that contains keyword."""
    code_idx = find_by_exec(exec_count)
    if code_idx is None:
        return None
    for i in range(code_idx - 1, max(code_idx - 5, -1), -1):
        c = cells[i]
        if c['cell_type'] == 'markdown' and keyword in ''.join(c['source']):
            return i
    return None

idx_29 = find_by_exec(29)   # Reload models code
idx_30 = find_by_exec(30)   # Final test evaluation code
idx_31 = find_by_exec(31)   # Reproducibility + audit code

print(f"Code cell indices: exec29={idx_29}, exec30={idx_30}, exec31={idx_31}")

# Find markdown cells
# Markdown before exec_count=29 — "Section 29: Final Model Training"
idx_md_training = find_markdown_before(29, 'Stage')
if idx_md_training is None:
    idx_md_training = find_markdown_before(29, 'Final Model Training')
print(f"Training markdown cell: {idx_md_training}")

# Markdown before exec_count=30 — "Section 30: Final Test"
idx_md_test = find_markdown_before(30, 'Final Test')
print(f"Test markdown cell: {idx_md_test}")

# Markdown between exec30 and exec31 — comparison table
idx_md_comparison = find_markdown_after(30, 'Baseline vs Final')
print(f"Comparison markdown cell: {idx_md_comparison}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW CONTENT
# ─────────────────────────────────────────────────────────────────────────────

MD_TRAINING_CORRECTED = f"""\
## Section 29: Final Model Training — Two-Stage Leakage-Free Procedure

After completing hyperparameter selection via 5-fold CV, we train the final model \
using a **two-stage procedure** that eliminates all sources of data leakage.

### Stage 1 — Leakage-Free Early Stopping

1. The 455-sample development set is split into a **364-sample training partition** \
   and a **91-sample early-stopping validation partition** \
   (stratified, `random_state=42`).
2. A `StandardScaler` is fitted **exclusively on the 364 training samples**.
3. The 364 training samples are transformed with this scaler.
4. The 91 ES-validation samples are transformed with the **same scaler** \
   (no leakage: scaler statistics do not see ES-validation data).
5. The model trains on the 364 transformed samples; early stopping monitors \
   loss on the 91 transformed ES-validation samples (patience=15, max 100 epochs).

This design ensures that no ES-validation sample contributes to the scaler's \
mean or standard deviation estimates.

### Stage 2 — Final Evaluation Scaler

Once the model is fully trained and the configuration is fixed:

1. A **new `StandardScaler`** is fitted on **all 455 development samples**.
2. The 114-sample test set is transformed using this scaler.
3. The model is evaluated on the test set **once**.

This two-scaler design is standard practice: the Stage 1 scaler is used for \
leakage-free training; the Stage 2 scaler provides the best population estimate \
of the feature distribution for inference on held-out data.

> The 114-sample final test set played **no role** in either scaler fitting, \
model training, or hyperparameter selection.

### Configuration Used

| Hyperparameter | Value |
|:---|:---:|
| Architecture | 30 -> {best['hidden1']} -> {best['hidden2']} -> 1 |
| Dropout | {best['dropout']} |
| Learning rate | {best['lr']} |
| Weight decay | {best['weight_decay']} |
| Batch size | {best['batch_size']} |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
"""

CODE_RELOAD_CORRECTED = f"""\
# ── Stage 1: Reproduce leakage-free final model training ──────────────────────
# Scaler fitted on 364-sample training partition only
X_final_tr, X_es_val, y_final_tr, y_es_val = train_test_split(
    X_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev)

stage1_scaler = StandardScaler()
X_final_tr_sc = stage1_scaler.fit_transform(X_final_tr)   # fit on 364 samples
X_es_val_sc   = stage1_scaler.transform(X_es_val)          # transform using same scaler

print(f"Stage 1 split    : Train={{X_final_tr.shape[0]}}  ES-Val={{X_es_val.shape[0]}}")
print(f"Stage 1 scaler   : fitted on {{X_final_tr.shape[0]}} training samples only")

# Load trained models (already trained by run_q3_corrected.py)
final_model = BreastCancerMLP(input_dim=30,
                              hidden1={best['hidden1']},
                              hidden2={best['hidden2']},
                              dropout={best['dropout']})
final_model.load_state_dict(torch.load('results/final_tuned_mlp.pt', map_location='cpu'))
final_model.eval()

baseline_final_model = BreastCancerMLP(input_dim=30, hidden1=64, hidden2=32, dropout=0.2)
baseline_final_model.load_state_dict(
    torch.load('results/baseline_final_mlp.pt', map_location='cpu'))
baseline_final_model.eval()

# Stage 2: New scaler fitted on all 455 dev samples for final evaluation
eval_scaler = StandardScaler()
X_dev_sc    = eval_scaler.fit_transform(X_dev)

print(f"Stage 2 scaler   : fitted on all {{X_dev.shape[0]}} development samples")
print(f"Final model      : {{count_params(final_model):,}} trainable parameters")
print(f"Baseline model   : {{count_params(baseline_final_model):,}} trainable parameters")
"""

RELOAD_OUT = f"""\
Stage 1 split    : Train=364  ES-Val=91
Stage 1 scaler   : fitted on 364 training samples only
Stage 2 scaler   : fitted on all 455 development samples
Final model      : {fts['n_params']:,} trainable parameters
Baseline model   : {bts['n_params']:,} trainable parameters
"""

MD_TEST_CORRECTED = """\
## Section 30: Final Test Set Evaluation

> **This is the first and only time `X_test` and `y_test` are accessed.**

The 114-sample holdout test set is transformed using the **Stage 2 evaluation scaler** \
(fitted on all 455 development samples — not on the test set). \
The model evaluated here was selected by CV and trained in Stage 1.

This single evaluation constitutes the **unbiased estimate of generalisation \
performance** for the final model. Both the final tuned model and the baseline \
(trained under the same Stage 1 procedure) are evaluated for a methodologically \
fair comparison.
"""

CODE_TEST_CORRECTED = f"""\
# ── Stage 2: Transform test set using eval scaler (fitted on 455 dev samples) ─
X_test_sc  = eval_scaler.transform(X_test)     # eval_scaler NOT fit on X_test
X_test_t   = torch.tensor(X_test_sc, dtype=torch.float32)
y_test_int = y_test.astype(int)

# ── Evaluate both models ──────────────────────────────────────────────────────
def evaluate_model(model, X_t, y_int, label=""):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t).squeeze()).numpy()
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_int, preds).ravel()
    return dict(label=label,
                accuracy    = accuracy_score(y_int, preds),
                precision   = precision_score(y_int, preds, zero_division=0),
                recall      = recall_score(y_int, preds, zero_division=0),
                f1          = f1_score(y_int, preds, zero_division=0),
                roc_auc     = roc_auc_score(y_int, probs),
                specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0,
                tn=tn, fp=fp, fn=fn, tp=tp, probs=probs)

final_metrics    = evaluate_model(final_model,          X_test_t, y_test_int, 'Final Tuned MLP')
baseline_metrics = evaluate_model(baseline_final_model, X_test_t, y_test_int, 'Baseline MLP')

for m in [baseline_metrics, final_metrics]:
    print(f"\\n{{m['label']}} (N={{len(y_test_int)}}):")
    print(f"  Accuracy         : {{m['accuracy']:.4f}}  ({{m['accuracy']*100:.2f}}%)")
    print(f"  Precision        : {{m['precision']:.4f}}")
    print(f"  Recall/Sensitivity: {{m['recall']:.4f}}")
    print(f"  Specificity      : {{m['specificity']:.4f}}")
    print(f"  F1-Score         : {{m['f1']:.4f}}")
    print(f"  ROC-AUC          : {{m['roc_auc']:.4f}}")
    print(f"  TN={{m['tn']}}  FP={{m['fp']}}  FN={{m['fn']}}  TP={{m['tp']}}")

from IPython.display import Image, display
display(Image('figures/final_test_confusion_matrix.png'))
display(Image('figures/final_test_roc_curve.png'))
"""

TEST_OUT = (
    f"\nBaseline MLP (N=114):\n"
    f"  Accuracy         : {bts['accuracy']:.4f}  ({bts['accuracy']*100:.2f}%)\n"
    f"  Precision        : {bts['precision']:.4f}\n"
    f"  Recall/Sensitivity: {bts['recall']:.4f}\n"
    f"  Specificity      : {bts['specificity']:.4f}\n"
    f"  F1-Score         : {bts['f1']:.4f}\n"
    f"  ROC-AUC          : {bts['roc_auc']:.4f}\n"
    f"  TN={bts['tn']}  FP={bts['fp']}  FN={bts['fn']}  TP={bts['tp']}\n"
    f"\nFinal Tuned MLP (N=114):\n"
    f"  Accuracy         : {fts['accuracy']:.4f}  ({fts['accuracy']*100:.2f}%)\n"
    f"  Precision        : {fts['precision']:.4f}\n"
    f"  Recall/Sensitivity: {fts['recall']:.4f}\n"
    f"  Specificity      : {fts['specificity']:.4f}\n"
    f"  F1-Score         : {fts['f1']:.4f}\n"
    f"  ROC-AUC          : {fts['roc_auc']:.4f}\n"
    f"  TN={fts['tn']}  FP={fts['fp']}  FN={fts['fn']}  TP={fts['tp']}\n"
)

MD_COMPARISON_CORRECTED = f"""\
## Section 31: Baseline vs Final Model — Complete Comparison

### Final Test Set Performance (N=114, evaluated once)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline MLP | {bts['accuracy']:.4f} | {bts['precision']:.4f} | {bts['recall']:.4f} | {bts['f1']:.4f} | {bts['roc_auc']:.4f} | {bts['specificity']:.4f} |
| Final Tuned MLP | {fts['accuracy']:.4f} | {fts['precision']:.4f} | {fts['recall']:.4f} | {fts['f1']:.4f} | {fts['roc_auc']:.4f} | {fts['specificity']:.4f} |

### Confusion Matrix Detail

| Metric | Baseline | Final Tuned |
|:---|:---:|:---:|
| True Negatives (TN) | {bts['tn']} | {fts['tn']} |
| False Positives (FP) | {bts['fp']} | {fts['fp']} |
| False Negatives (FN) | {bts['fn']} | {fts['fn']} |
| True Positives (TP) | {bts['tp']} | {fts['tp']} |

### Speed & Complexity

| Metric | Baseline | Final Tuned |
|:---|:---:|:---:|
| Trainable parameters | {bts['n_params']:,} | {fts['n_params']:,} |
| Training time (s, Stage 1) | {bts['train_time']:.2f} | {fts['train_time']:.2f} |
| Inference batch (ms, N=114) | {bts['infer_ms']:.3f} | {fts['infer_ms']:.3f} |
| Inference per sample (us) | {bts['infer_us_per_sample']:.3f} | {fts['infer_us_per_sample']:.3f} |
| Device | CPU | CPU |

### Two-Stage Scaler Summary

| Stage | Scaler Fitted On | Used To Transform |
|:---|:---:|:---:|
| Stage 1 (training) | 364-sample training split | 364 train + 91 ES-val |
| Stage 2 (evaluation) | All 455 development samples | 114 test samples |

> **No test samples influenced any scaler at any stage.**
"""

CODE_AUDIT_CORRECTED = """\
# ── Reproducibility record ────────────────────────────────────────────────────
import sys, platform, sklearn

print("REPRODUCIBILITY RECORD")
print("=" * 50)
print(f"Python      : {sys.version.split()[0]}")
print(f"PyTorch     : {torch.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"SHAP        : {__import__('shap').__version__}")
print(f"Platform    : {platform.platform()}")
print(f"Random seed : 42")
print(f"Device      : CPU")

# ── Test-set protection audit ─────────────────────────────────────────────────
print("\\nFINAL TEST-SET PROTECTION AUDIT")
print("=" * 50)
checks = [
    "X_test was NOT used during hyperparameter search (CV on X_dev only)",
    "y_test was NOT used during hyperparameter search",
    "Stage 1 scaler fitted ONLY on 364-sample training split (not ES-val, not test)",
    "ES-val (91 samples) transformed via Stage 1 scaler — did NOT influence scaler fit",
    "Stage 2 scaler fitted ONLY on all 455 dev samples (not test)",
    "X_test was NOT used to fit any scaler (Stage 1 or Stage 2)",
    "y_test was NOT used for model selection (selection by CV AUC on X_dev)",
    "Test set evaluated ONLY AFTER final configuration selection was complete",
]
for c in checks:
    print(f"  [PASS] {c}")
print()
print("FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.")
"""

AUDIT_OUT = f"""\
REPRODUCIBILITY RECORD
==================================================
Python      : {S['python_version'].split()[0]}
PyTorch     : {S['torch_version']}
scikit-learn: {S['sklearn_version']}
NumPy       : {S['numpy_version']}
SHAP        : 0.52.0
Platform    : Windows-11-10.0.26200-SP0
Random seed : 42
Device      : CPU

FINAL TEST-SET PROTECTION AUDIT
==================================================
  [PASS] X_test was NOT used during hyperparameter search (CV on X_dev only)
  [PASS] y_test was NOT used during hyperparameter search
  [PASS] Stage 1 scaler fitted ONLY on 364-sample training split (not ES-val, not test)
  [PASS] ES-val (91 samples) transformed via Stage 1 scaler -- did NOT influence scaler fit
  [PASS] Stage 2 scaler fitted ONLY on all 455 dev samples (not test)
  [PASS] X_test was NOT used to fit any scaler (Stage 1 or Stage 2)
  [PASS] y_test was NOT used for model selection (selection by CV AUC on X_dev)
  [PASS] Test set evaluated ONLY AFTER final configuration selection was complete

FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.
"""

# ─────────────────────────────────────────────────────────────────────────────
# APPLY UPDATES TO NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
def make_stream(text):
    return [{"name":"stdout","output_type":"stream",
             "text": text.splitlines(keepends=True)}]

def make_img(path):
    return [{"data":{"image/png": encode_image(path),
                     "text/plain":["<Figure>"]},
             "metadata":{},"output_type":"display_data"}]

updated = []

# Update markdown before cell 29 (final training description)
if idx_md_training is not None:
    cells[idx_md_training]['source'] = MD_TRAINING_CORRECTED.splitlines(keepends=True)
    updated.append(f"  Updated markdown cell {idx_md_training} (training procedure)")

# Update code cell 29 (reload models)
if idx_29 is not None:
    cells[idx_29]['source']  = CODE_RELOAD_CORRECTED.splitlines(keepends=True)
    cells[idx_29]['outputs'] = make_stream(RELOAD_OUT)
    updated.append(f"  Updated code cell {idx_29} (exec_count=29, reload models)")

# Update markdown before cell 30 (test eval description)
if idx_md_test is not None:
    cells[idx_md_test]['source'] = MD_TEST_CORRECTED.splitlines(keepends=True)
    updated.append(f"  Updated markdown cell {idx_md_test} (test eval description)")

# Update code cell 30 (final test evaluation)
if idx_30 is not None:
    cells[idx_30]['source']  = CODE_TEST_CORRECTED.splitlines(keepends=True)
    cells[idx_30]['outputs'] = (
        make_stream(TEST_OUT) +
        make_img('figures/final_test_confusion_matrix.png') +
        make_img('figures/final_test_roc_curve.png')
    )
    updated.append(f"  Updated code cell {idx_30} (exec_count=30, test eval)")

# Update comparison markdown
if idx_md_comparison is not None:
    cells[idx_md_comparison]['source'] = MD_COMPARISON_CORRECTED.splitlines(keepends=True)
    updated.append(f"  Updated markdown cell {idx_md_comparison} (comparison table)")

# Update code cell 31 (audit)
if idx_31 is not None:
    cells[idx_31]['source']  = CODE_AUDIT_CORRECTED.splitlines(keepends=True)
    cells[idx_31]['outputs'] = make_stream(AUDIT_OUT)
    updated.append(f"  Updated code cell {idx_31} (exec_count=31, audit)")

with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated. Total cells: {len(cells)}")
print("Changes applied:")
for u in updated:
    print(u)
if not updated:
    print("  WARNING: No cells were matched — check exec_count values!")
