"""
Inserts the AI-Use Statement markdown cell into hiring_test.ipynb.
"""
import json, uuid

def new_id():
    return uuid.uuid4().hex[:8]

AI_STATEMENT_MD = """\
---

## AI-Use Statement

AI tools were used during the preparation of this technical test.

**ChatGPT (OpenAI):** I used ChatGPT as an assistance tool for understanding and discussing machine-learning concepts, model interpretability concepts, methodological considerations, and for reviewing the organization and explanation of the experimental work. ChatGPT was also used to help with wording and presentation of technical explanations.

**Google Antigravity:** I used Google Antigravity as an AI-assisted development environment to help implement, execute, inspect, and organize the Python/Jupyter Notebook code used for the dataset analysis, model development, SHAP analysis, hyperparameter evaluation, visualization, and related experimental workflow.

I reviewed the generated code, outputs, methodology, and results and used the resulting notebook as the basis for the submitted answers. AI assistance was used as a development and explanatory aid; the submitted experimental results are those produced by the executed code in the notebook.
"""

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if AI-Use statement already present
exists = any('AI-Use Statement' in ''.join(c['source']) for c in nb['cells'])
if not exists:
    ai_cell = {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": AI_STATEMENT_MD.splitlines(keepends=True)
    }
    nb['cells'].append(ai_cell)
    with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Added AI-Use Statement cell. Total cells now: {len(nb['cells'])}")
else:
    print(f"AI-Use Statement already present. Total cells: {len(nb['cells'])}")
