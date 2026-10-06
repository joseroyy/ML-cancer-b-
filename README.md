# Computational Biomedicine Lab — Research Hiring Test

## Overview
This repository contains the initial data loading, exploratory inspection, schema mapping, and data-quality audit for the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset as part of a Computational Biomedicine Lab research-position evaluation.

---

## Dataset Source
**UCI Machine Learning Repository:**  
**Breast Cancer Wisconsin (Diagnostic)**  
[https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)

- **Dataset Creators:** Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin)
- **Donor:** Nick Street (November 1995)
- **Files Location:** `data/wdbc.data` (raw dataset) and `data/wdbc.names` (official attribute documentation)

---

## Dataset Characteristics & Biological Relevance
Features in this dataset are computed from digitized images of fine needle aspirates (FNA) of breast masses. They describe physical characteristics of cell nuclei present in the digitized image:

1. **Radius:** Mean of distances from center to points on the perimeter
2. **Texture:** Standard deviation of gray-scale values
3. **Perimeter:** Contour length of cell nuclei
4. **Area:** Enclosed nuclear area
5. **Smoothness:** Local variation in radius lengths
6. **Compactness:** $\frac{\text{perimeter}^2}{\text{area}} - 1.0$
7. **Concavity:** Severity of concave portions of the contour
8. **Concave Points:** Number of concave portions of the contour
9. **Symmetry:** Cellular symmetry
10. **Fractal Dimension:** "Coastline approximation" minus 1

The mean, standard error (SE), and worst (mean of the three largest values) of these 10 base characteristics were computed for each image, yielding **30 quantitative numerical features**.

---

## Data Summary & Initial Inspection Findings

| Dimension / Metric | Result |
| :--- | :--- |
| **Total Number of Samples (Rows)** | `569` |
| **Total Number of Attributes (Columns)** | `32` (1 ID + 1 Target + 30 Numerical Features) |
| **Target Variable** | `diagnosis` (`M` = Malignant, `B` = Benign) |
| **Class Distribution (Counts)** | Benign (B): `357`, Malignant (M): `212` |
| **Class Distribution (%)** | Benign (B): `62.74%`, Malignant (M): `37.26%` |
| **Missing Values** | `0` missing values across all 32 columns |
| **Duplicate Rows** | `0` duplicate rows |
| **ID Column Integrity** | `569` unique patient IDs (100% unique) |
| **Zero-Variance Features** | `0` constant features |
| **Data Transformation Status** | Untransformed raw state (no scaling, splitting, or modeling) |

### Key Data-Quality Observations
- **Zero Missingness:** Complete dataset without any missing attributes or NaN values.
- **Scale Disparities:** Features exhibit wide variance in scale ranges (e.g., `fractal_dimension_se` ranges from ~0.001 to 0.03, while `area_worst` ranges from 185.2 to 4254.0).
- **Biological Concavity Zeros:** 13 samples have exact `0.0` values across concavity features, representing nuclei with smooth convex contours without concave indentations.
- **Moderate Class Imbalance:** ~62.7% Benign vs ~37.3% Malignant.

---

## Notebook Structure
The analysis is implemented in `hiring_test.ipynb`:
1. **Title & Research Context:** Clinical background and suitability for computational biomedicine binary classification.
2. **Directory Inspection:** Identifying dataset files inside `data/`.
3. **Official UCI Schema Mapping:** Mapping `wdbc.names` definitions to 32 explicit column names.
4. **Structural Display:** Shape, first 5 rows, last 5 rows, column list, dtypes, missing values count, duplicate rows count.
5. **Target Distribution:** Counts, percentages, and class bar plot preserving original `M` and `B` labels.
6. **Descriptive Statistics:** Summary metrics across all 30 numerical features.
7. **Data Quality Audit:** Comprehensive checks for nulls, duplicates, zero-variance columns, unexpected labels, scale differences, and zero values.
8. **Research Summary:** Structured synthesis of findings.
