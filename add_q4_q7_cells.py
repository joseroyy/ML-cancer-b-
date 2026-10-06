"""
Adds Questions 4-7 to hiring_test.ipynb after Question 3 and before the AI-Use Statement.
Does not touch any Q1-Q3 code, metrics, outputs, figures, or the AI-Use Statement.
"""

import json, uuid

def new_id():
    return uuid.uuid4().hex[:8]

Q4_MD = """\
---

# Question 4 — Unintended Consequences of AI

One example that stands out to me is a healthcare algorithm that was found to underestimate the healthcare needs of Black patients because it used healthcare spending as a proxy for medical need. Since spending was also affected by unequal access to healthcare, the model could appear accurate while still producing unfair results.

This shows me that good model performance alone is not enough, especially in healthcare. Before deploying an AI system, I would look at how the target variable is defined, whether the data represents different groups properly, and whether errors affect some groups more than others.
"""

Q5_MD = """\
---

# Question 5 — Fairness in AI

To me, fairness in AI means making sure that a model does not systematically disadvantage a particular group because of the data or the way the model is designed. Bias can enter through the training data, missing information, labels, or even the way a problem is defined.

In Computational Biomedicine, I would check whether the model performs consistently across relevant patient groups. For example, in medical image analysis, I would compare sensitivity, specificity and false-negative rates across groups. A model should not be considered reliable just because its overall accuracy is high.
"""

Q6_MD = """\
---

# Question 6 — Resistant Non-Technical Stakeholder

If a non-technical stakeholder was resistant to using a model I developed, I would first try to understand why. Their concern could be about reliability, explainability, patient safety, privacy, or simply changing their existing workflow.

Instead of explaining the model architecture, I would show them practical results, including examples where the model was correct and where it made mistakes. I would also suggest testing it in a controlled setting first and collecting their feedback. This would make the discussion more about solving the actual problem than convincing them to accept the technology.
"""

Q7_MD = """\
---

# Question 7 — Collaborative Problem Solving

[I will add a real example from my project experience here.]
"""

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find index of AI-Use Statement cell
ai_idx = None
for i, c in enumerate(cells):
    if 'AI-Use Statement' in ''.join(c['source']):
        ai_idx = i
        break

print(f"Total cells before addition: {len(cells)}")
print(f"AI-Use Statement cell index: {ai_idx}")

# Check if Q4-Q7 already added
q4_exists = any('Question 4' in ''.join(c['source']) for c in cells)

if not q4_exists:
    q_cells = [
        {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": Q4_MD.splitlines(keepends=True)},
        {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": Q5_MD.splitlines(keepends=True)},
        {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": Q6_MD.splitlines(keepends=True)},
        {"cell_type": "markdown", "id": new_id(), "metadata": {}, "source": Q7_MD.splitlines(keepends=True)}
    ]
    
    if ai_idx is not None:
        # Insert before AI-Use Statement
        nb['cells'] = cells[:ai_idx] + q_cells + cells[ai_idx:]
    else:
        # Append to end
        nb['cells'].extend(q_cells)

    with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully added Questions 4-7. Total cells now: {len(nb['cells'])}")
else:
    print("Questions 4-7 already present in notebook.")
