# Iris Dataset Classification (CodeAlpha)

A **complete, end‑to‑end machine learning classification project** using the classic **Iris dataset**. This repository demonstrates a **disciplined, industry‑style ML workflow** — from exploratory data analysis (EDA) to baseline modeling, systematic hyperparameter tuning, error analysis, and honest final evaluation using a hold‑out test set.

---

## Project Overview

The goal of this project is to **predict the species of an Iris flower** — *Setosa, Versicolor, or Virginica* — using four physical measurements:

* Sepal length
* Sepal width
* Petal length
* Petal width

This is a **multi‑class classification problem** and serves as an excellent foundation for understanding real‑world supervised learning pipelines.

---

## Key Learning Objectives

* Understand **EDA for classification problems**
* Visualize **univariate, bivariate, and multivariate relationships**
* Build a **manual rule‑based baseline model**
* Implement a **proper train → validation → test workflow**
* Apply **cross‑validation and hyperparameter tuning**
* Perform **error analysis with visual diagnostics**
* Interpret **confusion matrices and generalization gaps**

---

## ⚙️ Environment & Libraries

**Language:** Python
**Core Libraries:**

* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit‑learn

The dataset is loaded directly from **scikit‑learn’s built‑in Iris dataset**, ensuring reproducibility.

---

##  Exploratory Data Analysis (EDA)

The EDA phase includes:

* **Univariate analysis** (histograms)
* **Bivariate analysis** (feature vs target relationships)
* **Multivariate analysis** (pair plots)

###  Key EDA Insights

* **Petal length & petal width** are the most predictive features
* *Setosa* is perfectly separable from the other species
* Most ambiguity occurs between *Versicolor* and *Virginica*

These insights directly motivate the modeling strategy.

---

## Baseline Models

### Random Guess Baseline

* Accuracy: **33.3%** (balanced 3‑class problem)

### Manual Rule‑Based Model

A simple threshold rule using **petal length only**:

* Test Accuracy: **~95%**

This confirms the dataset contains a **very strong signal**.

---

## Machine Learning Model

### Logistic Regression (Multinomial)

Implemented using a **Pipeline**:

* `StandardScaler`
* `LogisticRegression`

### Model Selection Strategy

* Train / Validation / Test split
* 5‑fold cross‑validation
* Manual + GridSearch hyperparameter tuning for **C**

---

## Hyperparameter Tuning

* **Coarse grid search** → identify promising region
* **Fine grid search** → pinpoint optimal value

**Best parameter:**

```
C = 0.5
```

---

## Final Results (Hold‑Out Test Set)

| Metric             | Value     |
| ------------------ | --------- |
| Accuracy           | **92.1%** |
| CV Accuracy        | **96.4%** |
| Generalization Gap | **~4.3%** |

### Confusion Matrix Insights

* *Setosa* → 100% accuracy
* Errors occur only between *Versicolor* and *Virginica*
* Misclassifications lie near natural decision boundaries

---

##  Error Analysis

* Used `cross_val_predict` for **honest per‑sample predictions**
* Visualized misclassified points in petal feature space
* Confirmed that errors occur in **overlapping regions**, not due to model flaws

This moves evaluation **beyond accuracy** into interpretability.

---

##  Key Conclusions

* Iris dataset contains a **strong predictive signal**
* Simple rules already perform surprisingly well
* Logistic Regression achieves **high accuracy with minimal complexity**
* Slight overfitting is expected and well‑quantified

---

##  Next Steps

* Compare Logistic Regression with:

  * Random Forest
  * Support Vector Machines (SVM)
* Apply **nested cross‑validation**
* Extend workflow to larger, noisier datasets

---

##  How to Run

1. Open the notebook in **Google Colab** or locally
2. Run cells top‑to‑bottom
3. All results are fully reproducible (fixed random seeds)
