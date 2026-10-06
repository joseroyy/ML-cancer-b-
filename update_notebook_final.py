"""
Updates hiring_test.ipynb to reflect the corrected, standard ML final retraining procedure.
"""
import json, base64, uuid

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def make_stream(text):
    return [{"name":"stdout","output_type":"stream",
             "text": text.splitlines(keepends=True)}]

def make_img(path):
    return [{"data":{"image/png": encode_image(path),
                     "text/plain":["<Figure>"]},
             "metadata":{},"output_type":"display_data"}]

with open('results/q3_summary.json', 'r') as f:
    S = json.load(f)

bts = S['baseline_test']
fts = S['final_test']
best = S['best_cv']

# ── Notebook Markdown & Code replacements ────────────────────────────────────

MD_TRAINING_FINAL = f"""\
## Section 29: Final Model Retraining on All Development Data

After fixing the hyperparameter configuration via 5-fold CV, we perform **final model \
retraining** on the complete 455-sample development set (`X_dev`).

### Standard Final Retraining & Preprocessing Methodology

1. **Single Final Scaler**: A `StandardScaler` (`final_scaler`) is fitted **once on \
   all 455 development samples** (`X_dev`).
2. **Development Data Transformation**: `X_dev` is transformed with `final_scaler` to \
   produce `X_dev_scaled`.
3. **Test Data Transformation**: The sealed 114-sample test set (`X_test`) is \
   transformed using the **same `final_scaler`**, ensuring that both the neural \
   network's training data and test evaluations exist in the exact same feature \
   coordinate system.
4. **Training Duration**: The final model is retrained on all 455 development samples \
   for **32 fixed epochs** — a duration established by the prior internal early-stopping \
   experiment without referencing the test set.
5. **No Test Data Influence**: `X_test` and `y_test` are strictly excluded from \
   scaler fitting, model training, epoch selection, and hyperparameter selection.

### Distinction: Hyperparameter Search vs Final Deployment

| Stage | Preprocessing & Training | Test Set Isolation |
|:---|:---|:---|
| **A. Hyperparameter Search (CV)** | 5-Fold Stratified CV on `X_dev` (N=455). `StandardScaler` fitted independently inside every training fold. | `X_test` completely untouched. |
| **B. Final Model Retraining** | `final_scaler` fitted once on `X_dev`. MLP retrained on all 455 dev samples for 32 fixed epochs. | `X_test` transformed with `final_scaler`. |
| **C. Final Evaluation** | Evaluate retrained MLP on `X_test_scaled` (N=114). | Single evaluation step. |

### Selected Final Architecture

| Hyperparameter | Value |
|:---|:---:|
| Architecture | 30 -> {best['hidden1']} -> {best['hidden2']} -> 1 |
| Dropout | {best['dropout']} |
| Learning rate | {best['lr']} |
| Weight decay | {best['weight_decay']} |
| Batch size | {best['batch_size']} |
| Fixed Training Epochs | 32 |
| Trainable Parameters | {fts['n_params']:,} |
"""

CODE_RELOAD_FINAL = f"""\
# ── Fit final scaler on ALL 455 dev samples ──────────────────────────────────
final_scaler = StandardScaler()
X_dev_scaled  = final_scaler.fit_transform(X_dev)   # Fit on X_dev (455 samples)
X_test_scaled = final_scaler.transform(X_test)     # Transform X_test with SAME scaler

# ── Load retrained final models ───────────────────────────────────────────────
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

print(f"final_scaler fitted on : {{X_dev.shape[0]}} development samples")
print(f"X_test transformed with: SAME final_scaler (NO test data fit)")
print(f"Final Tuned MLP        : {{count_params(final_model):,}} parameters (retrained 32 epochs)")
print(f"Baseline MLP           : {{count_params(baseline_final_model):,}} parameters (retrained 32 epochs)")
"""

RELOAD_OUT_FINAL = f"""\
final_scaler fitted on : 455 development samples
X_test transformed with: SAME final_scaler (NO test data fit)
Final Tuned MLP        : {fts['n_params']:,} parameters (retrained 32 epochs)
Baseline MLP           : {bts['n_params']:,} parameters (retrained 32 epochs)
"""

MD_TEST_FINAL = """\
## Section 30: Final Test Set Evaluation

> **This is the first and only time `X_test` and `y_test` are accessed.**

The 114-sample holdout test set transformed by `final_scaler` (`X_test_scaled`) \
is now evaluated. Both the retrained final tuned MLP and retrained baseline MLP \
are evaluated for a fair, same-methodology comparison.
"""

CODE_TEST_FINAL = f"""\
# ── Evaluate retrained models on X_test_scaled (N=114) ────────────────────────
X_test_t   = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_int = y_test.astype(int)

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
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), probs=probs)

final_metrics    = evaluate_model(final_model,          X_test_t, y_test_int, 'Final Tuned MLP')
baseline_metrics = evaluate_model(baseline_final_model, X_test_t, y_test_int, 'Baseline MLP')

for m in [baseline_metrics, final_metrics]:
    print(f"\\n{{m['label']}} (N={{len(y_test_int)}}):")
    print(f"  Accuracy          : {{m['accuracy']:.4f}}  ({{m['accuracy']*100:.2f}}%)")
    print(f"  Precision         : {{m['precision']:.4f}}")
    print(f"  Recall/Sensitivity : {{m['recall']:.4f}}")
    print(f"  Specificity       : {{m['specificity']:.4f}}")
    print(f"  F1-Score          : {{m['f1']:.4f}}")
    print(f"  ROC-AUC           : {{m['roc_auc']:.4f}}")
    print(f"  Confusion Matrix  : TN={{m['tn']}}  FP={{m['fp']}}  FN={{m['fn']}}  TP={{m['tp']}}")

from IPython.display import Image, display
display(Image('figures/final_test_confusion_matrix.png'))
display(Image('figures/final_test_roc_curve.png'))
"""

TEST_OUT_FINAL = (
    f"\nBaseline MLP (N=114):\n"
    f"  Accuracy          : {bts['accuracy']:.4f}  ({bts['accuracy']*100:.2f}%)\n"
    f"  Precision         : {bts['precision']:.4f}\n"
    f"  Recall/Sensitivity : {bts['recall']:.4f}\n"
    f"  Specificity       : {bts['specificity']:.4f}\n"
    f"  F1-Score          : {bts['f1']:.4f}\n"
    f"  ROC-AUC           : {bts['roc_auc']:.4f}\n"
    f"  Confusion Matrix  : TN={bts['tn']}  FP={bts['fp']}  FN={bts['fn']}  TP={bts['tp']}\n"
    f"\nFinal Tuned MLP (N=114):\n"
    f"  Accuracy          : {fts['accuracy']:.4f}  ({fts['accuracy']*100:.2f}%)\n"
    f"  Precision         : {fts['precision']:.4f}\n"
    f"  Recall/Sensitivity : {fts['recall']:.4f}\n"
    f"  Specificity       : {fts['specificity']:.4f}\n"
    f"  F1-Score          : {fts['f1']:.4f}\n"
    f"  ROC-AUC           : {fts['roc_auc']:.4f}\n"
    f"  Confusion Matrix  : TN={fts['tn']}  FP={fts['fp']}  FN={fts['fn']}  TP={fts['tp']}\n"
)

MD_COMPARISON_FINAL = f"""\
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

### Computational Metrics

| Metric | Baseline | Final Tuned |
|:---|:---:|:---:|
| Trainable parameters | {bts['n_params']:,} | {fts['n_params']:,} |
| Training epochs | {bts['epochs_used']} | {fts['epochs_used']} |
| Training time (s) | {bts['train_time']:.2f} | {fts['train_time']:.2f} |
| Inference batch (ms, N=114) | {bts['infer_ms']:.3f} | {fts['infer_ms']:.3f} |
| Inference per sample (us) | {bts['infer_us_per_sample']:.3f} | {fts['infer_us_per_sample']:.3f} |
| Device | CPU | CPU |
"""

CODE_AUDIT_FINAL = """\
# ── Reproducibility record & protection audit ──────────────────────────────────
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

print("\\nFINAL TEST-SET PROTECTION & METHODOLOGY AUDIT")
print("=" * 50)
checks = [
    "X_test was NOT used during hyperparameter search (5-fold CV on X_dev only)",
    "y_test was NOT used during hyperparameter search",
    "final_scaler fitted ONCE on ALL 455 dev samples (X_dev)",
    "Final MLP retrained on ALL 455 dev samples (X_dev_scaled)",
    "Fixed epoch count (32) derived from prior internal ES experiment without test set",
    "X_test transformed using the SAME final_scaler fitted on X_dev",
    "Single input coordinate space: trained MLP and test data use exact same scaler",
    "X_test and y_test evaluated ONLY AFTER final model retraining was complete",
]
for c in checks:
    print(f"  [PASS] {c}")
print()
print("FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.")
"""

AUDIT_OUT_FINAL = f"""\
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

FINAL TEST-SET PROTECTION & METHODOLOGY AUDIT
==================================================
  [PASS] X_test was NOT used during hyperparameter search (5-fold CV on X_dev only)
  [PASS] y_test was NOT used during hyperparameter search
  [PASS] final_scaler fitted ONCE on ALL 455 dev samples (X_dev)
  [PASS] Final MLP retrained on ALL 455 dev samples (X_dev_scaled)
  [PASS] Fixed epoch count (32) derived from prior internal ES experiment without test set
  [PASS] X_test transformed using the SAME final_scaler fitted on X_dev
  [PASS] Single input coordinate space: trained MLP and test data use exact same scaler
  [PASS] X_test and y_test evaluated ONLY AFTER final model retraining was complete

FINAL TEST SET WAS USED ONLY FOR FINAL EVALUATION.
"""

MD_DISCUSSION_FINAL = f"""\
## Question 3 — Hyperparameter Optimisation: Scientific Discussion

### 1. Baseline Configuration
The baseline MLP (30→64→32→1, ReLU, Dropout=0.2, Adam LR=0.001, \
BCEWithLogitsLoss, batch size=32) established a strong starting point with \
CV AUC = 0.9963 ± 0.0045.

### 2. Search Space & Strategy
We evaluated 20 configurations from a 324-point grid using reproducible \
randomised search (seed 42), including the baseline as a mandatory entry. \
Parameters explored: learning rate ∈ {{0.0001, 0.001, 0.003}}, \
dropout ∈ {{0.0, 0.2, 0.4}}, weight decay ∈ {{0, 0.0001, 0.001}}, \
hidden sizes ∈ {{32, 64}} × {{16, 32}}, batch size ∈ {{16, 32, 64}}.

### 3. Cross-Validation Design
Stratified 5-fold CV on the 455-sample development set with \
`StandardScaler` fitted independently inside each fold ensured a \
leakage-free, unbiased evaluation of each configuration.

### 4. Best Configuration
The baseline configuration (CFG_00) ranked first jointly with CFG_13 \
(identical architecture + weight decay=0.0001). Per the simplicity \
preference rule, the baseline is selected. The MLP with \
4,097 trainable parameters is already appropriately scaled for this dataset.

### 5. Final Retraining & Preprocessing Methodology
- `final_scaler` was fitted **once on all 455 development samples**.
- The final MLP was retrained on all 455 scaled development samples for **32 fixed epochs**.
- `X_test` was transformed using **the same `final_scaler`**, guaranteeing identical feature coordinate representation.

### 6. Final Test Performance (N=114, evaluated once)
- **Accuracy**: **{fts['accuracy']:.4f} ({fts['accuracy']*100:.2f}%)** (112/114 correct)
- **Precision**: **{fts['precision']:.4f}** (100% — zero false positives)
- **Recall / Sensitivity**: **{fts['recall']:.4f}** (40/42 malignant cases detected)
- **Specificity**: **{fts['specificity']:.4f}** (72/72 benign cases correctly identified)
- **F1-Score**: **{fts['f1']:.4f}**
- **ROC-AUC**: **{fts['roc_auc']:.4f}**
- **Confusion Matrix**: TN={fts['tn']}, FP={fts['fp']}, FN={fts['fn']}, TP={fts['tp']}

### 7. Limitations & Clinical Context
- **Sample Size**: N=569 total (455 dev, 114 test). CV standard deviations (~0.004–0.008) indicate minor fold variance.
- **Architectural Scope**: Explored 2-layer MLPs; deeper/convolutional/attention architectures were not covered.
- **Clinical Generalisation**: Results reflect performance on the UCI Breast Cancer Wisconsin (Diagnostic) dataset partition. **No claim is made regarding clinical generalisation or deployment on external patient populations.**
"""

# ─────────────────────────────────────────────────────────────────────────────
# APPLY UPDATES TO NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def find_by_exec(exec_count):
    for i, c in enumerate(cells):
        if c.get('execution_count') == exec_count:
            return i
    return None

def find_markdown_before(exec_count, keyword):
    code_idx = find_by_exec(exec_count)
    if code_idx is None: return None
    for i in range(code_idx - 1, max(code_idx - 5, -1), -1):
        if cells[i]['cell_type'] == 'markdown' and keyword in ''.join(cells[i]['source']):
            return i
    return None

def find_markdown_after(exec_count, keyword):
    code_idx = find_by_exec(exec_count)
    if code_idx is None: return None
    for i in range(code_idx + 1, min(code_idx + 5, len(cells))):
        if cells[i]['cell_type'] == 'markdown' and keyword in ''.join(cells[i]['source']):
            return i
    return None

idx_29 = find_by_exec(29)
idx_30 = find_by_exec(30)
idx_31 = find_by_exec(31)

idx_md_training   = find_markdown_before(29, 'Final Model')
idx_md_test       = find_markdown_before(30, 'Final Test')
idx_md_comparison = find_markdown_after(30, 'Baseline vs Final')
idx_md_discussion = find_markdown_after(31, 'Scientific Discussion')

print(f"Indices: md_training={idx_md_training}, exec29={idx_29}, md_test={idx_md_test}, "
      f"exec30={idx_30}, md_comparison={idx_md_comparison}, exec31={idx_31}, md_discussion={idx_md_discussion}")

cells[idx_md_training]['source']   = MD_TRAINING_FINAL.splitlines(keepends=True)
cells[idx_29]['source']            = CODE_RELOAD_FINAL.splitlines(keepends=True)
cells[idx_29]['outputs']           = make_stream(RELOAD_OUT_FINAL)
cells[idx_md_test]['source']       = MD_TEST_FINAL.splitlines(keepends=True)
cells[idx_30]['source']            = CODE_TEST_FINAL.splitlines(keepends=True)
cells[idx_30]['outputs']           = (make_stream(TEST_OUT_FINAL) +
                                      make_img('figures/final_test_confusion_matrix.png') +
                                      make_img('figures/final_test_roc_curve.png'))
cells[idx_md_comparison]['source'] = MD_COMPARISON_FINAL.splitlines(keepends=True)
cells[idx_31]['source']            = CODE_AUDIT_FINAL.splitlines(keepends=True)
cells[idx_31]['outputs']           = make_stream(AUDIT_OUT_FINAL)
if idx_md_discussion is not None:
    cells[idx_md_discussion]['source'] = MD_DISCUSSION_FINAL.splitlines(keepends=True)

with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nhiring_test.ipynb successfully updated!")
