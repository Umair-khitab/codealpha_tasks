# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 09:29:09 2026

@author: khita
"""

# iris_classification.py
# Task 1: Iris Flower Classification
# CodeAlpha Data Science Internship
# This script trains a machine learning model
# to classify Iris flowers into three species
# using their physical measurements.
# Step 1: Import all required libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Step 2: Load the Iris dataset

# Scikit-learn already provides the Iris dataset,
# so we don’t need to download it manually.

iris = load_iris()

# X contains flower measurements (features)
# y contains flower species labels (target)
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names



# Step 3: Create a DataFrame for analysis

# Converting data into a Pandas DataFrame
# makes it easier to explore and visualize.

df = pd.DataFrame(X, columns=feature_names)
df["species"] = y

print("First 5 rows of the dataset:")
print(df.head())


# Step 4: Understand the dataset

print("\nDataset information:")
print(df.info())

print("\nChecking for missing values:")
print(df.isnull().sum())



# Step 5: Visualize the data
# Pairplot helps us see relationships between
# different features and how species differ.

sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()



# Step 6: Split data into training & testing
# 80% data for training, 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Step 7: Train the machine learning model
# Random Forest is powerful and works very well
# for classification problems like this.

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Step 8: Make predictions on test data
y_pred = model.predict(X_test)



# Step 9: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))



# Step 10: Display confusion matrix
# Confusion matrix shows correct vs incorrect predictions

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()
