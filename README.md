# 🦠 Breast Cancer Classification using Logistic Regression

An end-to-end Machine Learning pipeline predicting whether a breast mass is **Malignant (M)** or **Benign (B)** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 📌 Overview
Early diagnosis of breast cancer drastically improves survival rates. This project implements a **Logistic Regression** model optimized for high **Recall (Sensitivity)** to minimize False Negatives (missing malignant diagnoses).

## 🛠️ Machine Learning Pipeline
1. **Data Cleaning:** Removed non-predictive features (`id`, `Unnamed: 32`).
2. **Exploratory Data Analysis (EDA):** Visualized feature distributions and correlation matrices using `Seaborn`.
3. **Feature Scaling:** Applied `StandardScaler` fitted *only* on training data to prevent data leakage.
4. **Model Training:** Fitted a `LogisticRegression` classifier on normalized features.
5. **Threshold Tuning:** Analyzed decision thresholds ($0.3 - 0.7$) to optimize for patient diagnostic safety (maximizing Recall).

## 📊 Key Results
- **Model:** Logistic Regression
- **Primary Metric Focus:** Recall / Sensitivity (Malignant Class)
- **Scaling:** `StandardScaler` applied to all continuous continuous features.

