# codealpha_tasks
# 🌸 Iris Flower Classification

## 📌 Overview
This project is part of the **CodeAlpha Data Science Internship**.  
It demonstrates how to train and evaluate a machine learning model to classify **Iris flowers** into three species (*Setosa, Versicolor, Virginica*) using their physical measurements.

The script uses the **Random Forest Classifier** from scikit‑learn and includes:
- Data exploration
- Visualization
- Model training
- Prediction
- Evaluation

---

## 📂 File Information
- **Filename:** `iris_classification.py`  
- **Author:** khita  
- **Created On:** Jan 13, 2026  
- **Language:** Python 3  

---

## ⚙️ Workflow

### 1. Import Libraries
- `pandas` – data handling  
- `matplotlib`, `seaborn` – visualization  
- `scikit-learn` – dataset, model training, evaluation  

### 2. Load Dataset
- Iris dataset loaded from `sklearn.datasets`.  
- Features: sepal length, sepal width, petal length, petal width.  
- Target: species labels.

### 3. Create DataFrame
- Convert dataset into Pandas DataFrame.  
- Add `species` column for target labels.

### 4. Explore Dataset
- Display first 5 rows.  
- Show dataset info (data types, non‑null counts).  
- Check for missing values.

### 5. Visualize Data
- **Pairplot** using Seaborn to visualize feature relationships and species distribution.

### 6. Split Dataset
- Train/test split: 80% training, 20% testing.

### 7. Train Model
- Random Forest Classifier (`n_estimators=100`).  
- Train on training data.  
- Calculate and plot feature importance.

### 8. Make Predictions
- Predictions on test data.  
- Classify a custom flower input (`[5.1, 3.5, 1.4, 0.2]`).

### 9. Evaluate Model
- Accuracy score.  
- Classification report (precision, recall, F1‑score).  
- Confusion matrix heatmap.

---

## 📊 Outputs
- Pairplot of dataset.  
- Feature importance bar chart.  
- Prediction example for custom input.  
- Accuracy and classification report.  
- Confusion matrix heatmap.

---

## ✅ Key Notes
- Iris dataset is **clean and well‑separated**, so high accuracy is expected.  
- Random Forest is robust and interpretable.  
- Demonstrates a complete ML pipeline: **exploration → visualization → training → prediction → evaluation**.

---

## 🚀 How to Run
1. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
