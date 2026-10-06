"""
Updates Question 7 in hiring_test.ipynb with the user's specific response.
Replaces the placeholder. Leaves all other cells intact.
"""

import json

Q7_NEW_MD = """\
---

# Question 7 — Collaborative Problem Solving

During my MSc project, I worked in a team of four on a common project. I divided the work into individual tasks, but some of the assigned tasks were not completed as planned. As the project deadline approached, I took responsibility for the main coding and implementation work to make sure we could complete the project on time.

I also reassigned the remaining work so that my teammates could contribute to the project documentation and report preparation. This experience taught me that teamwork sometimes requires adapting the division of responsibilities based on the situation, while still making sure everyone has a role in completing the final work.
"""

with open('hiring_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False
for c in nb['cells']:
    if c['cell_type'] == 'markdown' and 'Question 7' in ''.join(c['source']):
        c['source'] = Q7_NEW_MD.splitlines(keepends=True)
        updated = True
        break

if updated:
    with open('hiring_test.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Question 7 updated successfully!")
else:
    print("WARNING: Question 7 cell not found!")
